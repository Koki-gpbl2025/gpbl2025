# -*- coding: utf-8 -*-
"""
Two-phase dialog (CHAT -> one-shot CHECK) with configurable health items — EN version
- Do N turns of small talk (CHAT), then explicitly announce the check and run CHECK
- CHECK asks only the configured items, each: main question once + one clarification if needed
- Health-topic guard applies only to the configured items during CHAT (no ad-hoc health Qs)
- Extraction is JSON-based via LLM, with English regex fallbacks (yes/no/negations)

Deps:
  pip install google-genai google-cloud-speech google-cloud-texttospeech sounddevice pynput
"""

import google.generativeai as genai
from google.generativeai import types       
from google.api_core import exceptions      

from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts

import argparse, os, sys, queue, time, json, tempfile, wave, threading, random, re, unicodedata
from enum import Enum

# ========= Auth =========
EMBEDDED_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBEDDED_GCLOUD_JSON = ""

# ========= Model =========
MODEL = "gemini-2.5-flash"

SYS_CORE = """You are a companion robot for older adults.
- Keep responses short, polite, and use simple wording; one fact per turn.
- Always end with a brief question to keep the conversation going.
- Do not diagnose or prescribe. If concerns arise, encourage contacting family or seeing a clinician (only in CHECK)."""

POLICY_CHAT = """
# Phase: CHAT (small talk)
- Do NOT introduce new questions about the configured health topics (sleep, meds, etc.).
- Do NOT dig into health. Keep it to small talk only.
- 1–2 sentences, end with a light small-talk question.
"""

POLICY_CHECK = """
# Phase: CHECK (health check)
- For each configured card: ask the fixed question once, then at most one clarification if needed.
- Do not add extra questions.
"""

# ========= Health item registry (configurable) =========
# key -> id/prompt/type + labels (used in the health guard)
TOPIC_REGISTRY = {
    "sleep_hours": {
        "id": "sleep", "prompt": "About how many hours did you sleep last night?", "type": "number",
        "labels": ["sleep", "slept", "hours", "insomnia", "rest"]
    },
    "meal_morning": {
        "id": "meal", "prompt": "Did you have breakfast today?", "type": "bool",
        "labels": ["meal", "breakfast", "ate", "eating", "skip breakfast"]
    },
    "med_taken": {
        "id": "med", "prompt": "Did you take your medicine today?", "type": "bool",
        "labels": ["medicine", "meds", "pill", "took meds", "missed meds", "dose"]
    },
    "pain": {
        "id": "pain", "prompt": "Are you having any pain?", "type": "bool",
        "labels": ["pain", "ache", "hurt", "headache", "stomachache", "back pain"]
    },
    "dizzy": {
        "id": "dizzy", "prompt": "Have you felt dizzy or lightheaded recently?", "type": "bool",
        "labels": ["dizzy", "lightheaded", "light-headed", "vertigo"]
    },
    "mood_1to5": {
        "id": "mood", "prompt": "How is your energy or mood on a 1–5 scale?", "type": "number",
        "labels": ["mood", "energy", "feeling", "scale"]
    },
}

# ========= Normalization & heuristics (EN) =========
def normalize_en(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    return s.strip().lower()

# Generic neg/pos phrases
NEG_PAT = re.compile(
    r"\b(?:no|nope|none|not really|don'?t|didn'?t|haven'?t|never|nothing|did not|do not|have not|"
    r"missed|skipped)\b",
    re.I,
)
POS_PAT = re.compile(
    r"\b(?:yes|yeah|yep|a bit|a little|some|somewhat|slightly|"
    r"headache|pain|ache|hurts?|dizzy|light[- ]?headed|took|ate|had)\b",
    re.I,
)

def bool_heuristic(answer: str) -> bool | None:
    a = normalize_en(answer)
    # Avoid "not bad" → positive; simple guard
    if re.search(r"\bnot bad\b", a): return True
    if POS_PAT.search(a) and not re.search(r"\bnot\b", a): return True
    if NEG_PAT.search(a): return False
    return None

def card_specific_bool(card_key: str, answer: str) -> bool | None:
    a = normalize_en(answer)
    if card_key == "med_taken":
        if re.search(r"\b(missed|forgot|skip+ed|didn'?t take|have(?: not)? taken)\b", a): return False
        if re.search(r"\b(took|have taken|did take|took my meds?|took my pill)\b", a): return True
    if card_key == "meal_morning":
        if re.search(r"\b(skip+ed (?:breakfast|meal)|didn'?t eat|no breakfast|no meal)\b", a): return False
        if re.search(r"\b(ate|had (?:breakfast|a meal)|i had breakfast)\b", a): return True
    # pain/dizzy rely on generic POS/NEG
    return None

# ========= Dynamic CFGs (extract / guard / chat) =========
def build_extract_general_cfg(enabled_keys: list[str]) -> types.GenerateContentConfig:
    # Only include selected keys
    parts = []
    for k in enabled_keys:
        t = TOPIC_REGISTRY[k]["type"]
        if t == "number":
            parts.append(f"\"{k}\": number|null")
        else:
            parts.append(f"\"{k}\": true|false|null")
    schema = "{ " + ", ".join(parts) + " }"
    sys_text = f"Extract health info from the English utterance. Return JSON only with this schema:\n{schema}"
    return types.GenerateContentConfig(
        system_instruction=sys_text,
        temperature=0.0,
        response_mime_type="application/json",
        max_output_tokens=120,
    )

def build_extract_focused_cfg(key: str, card_prompt: str) -> types.GenerateContentConfig:
    examples = (
        "EXAMPLES:\n"
        'Q: Are you having any pain? / A: no → {"pain": false}\n'
        'Q: Are you having any pain? / A: my head hurts a little → {"pain": true}\n'
        'Q: Have you felt dizzy recently? / A: not really → {"dizzy": false}\n'
        'Q: Did you take your medicine today? / A: yes → {"med_taken": true}\n'
        'Q: About how many hours did you sleep last night? / A: around 6 → {"sleep_hours": 6}\n'
    )
    rules = (
        "RULES:\n"
        "- Negations (no, not really, didn’t, haven’t, missed, skipped) → false for booleans.\n"
        "- Affirmations (yes, a bit, some, hurts, dizzy, took, ate) → true for booleans.\n"
        "- If a number is present, return it (e.g., 6, 7.5) for numeric fields.\n"
        "- Prefer true/false over null when possible.\n"
    )
    sys_text = (
        f"From the following English Q/A, return ONLY the {key} value as JSON. "
        f'Use key name "{key}" exactly. Other keys must NOT be included.\n'
        f"{examples}{rules}\n"
        f"- Question: {card_prompt}\n- Output example: {{\"{key}\": value}}\n"
    )
    return types.GenerateContentConfig(
        system_instruction=sys_text,
        temperature=0.0,
        response_mime_type="application/json",
        max_output_tokens=120,
    )

def build_guard_cfg(enabled_keys: list[str]) -> types.GenerateContentConfig:
    vocab = []
    for k in enabled_keys:
        labels = TOPIC_REGISTRY[k]["labels"]
        vocab.append(f"- {k}: " + " / ".join(labels))
    vocab_text = "\n".join(vocab) if vocab else "- (none)"
    sys_text = (
        "Decide whether the following English text mentions any of the TARGET topics, "
        "or asks a question about those topics. Return JSON only.\n"
        "TARGET topics:\n" + vocab_text + "\n"
        'Output: {"is_topic": true|false, "is_question": true|false, "reason": "<short reason>"}'
    )
    return types.GenerateContentConfig(
        system_instruction=sys_text,
        temperature=0.0,
        response_mime_type="application/json",
        max_output_tokens=100,
    )

def build_cfg_for_chat(state_json: str, strict: bool=False) -> types.GenerateContentConfig:
    policy = POLICY_CHAT
    if strict:
        policy += "\n- Do not include any vocabulary related to the TARGET topics."
    sys_text = f"{SYS_CORE}\n{policy}\n\n# Session state (for reference; do NOT echo it):\n{state_json}"
    return types.GenerateContentConfig(
        system_instruction=sys_text,
        temperature=0.25,
        max_output_tokens=200,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

# ========= LLM calls =========
def llm_extract_general(utterance: str, EXTRACT_GENERAL_CFG):
    try:
        utterance_n = normalize_en(utterance)
        r = MODEL_CLIENT.generate_content(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=utterance_n)])],
            config=EXTRACT_GENERAL_CFG
        )
        data = json.loads(r.text or "{}")
        return normalize_extract(data)
    except Exception as e:
        print("LLM extract error:", e)
        return {}

def llm_extract_focused(utterance: str, EXTRACT_FOCUSED_CFG):
    try:
        utterance_n = normalize_en(utterance)
        r = MODEL_CLIENT.generate_content(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=utterance_n)])],
            config=EXTRACT_FOCUSED_CFG
        )
        data = json.loads(r.text or "{}")
        return normalize_extract(data)
    except Exception as e:
        # optional: print("LLM focused extract error:", e)
        return {}


def guard_flags(utterance: str, GUARD_CFG):
    try:
        utterance_n = normalize_en(utterance)
        r = MODEL_CLIENT.generate_content(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=utterance_n)])],
            config=GUARD_CFG
        )
        data = json.loads(r.text or "{}")
        return data
    except Exception as e:
        # optional: print("Guard check error:", e)
        return {}


def gen_chat_with_guard(utterance: str, CHAT_CFG, GUARD_CFG):
    try:
        utterance_n = normalize_en(utterance)
        # Step 1: run guard
        guard_r = MODEL_CLIENT.generate_content(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=utterance_n)])],
            config=GUARD_CFG
        )
        guard_data = json.loads(guard_r.text or "{}")

        # If guard flags unsafe, return info only
        if guard_data.get("blocked", False):
            return {
                "reply": "⚠️ Content blocked by guardrails",
                "guard": guard_data
            }

        # Step 2: run main chat model
        chat_r = MODEL_CLIENT.generate_content(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=utterance_n)])],
            config=CHAT_CFG
        )
        return {
            "reply": chat_r.text or "",
            "guard": guard_data
        }
    except Exception as e:
        return {
            "reply": "❌ Error while generating response",
            "guard": {}
        }


# ========= Utilities =========
def normalize_extract(d: dict) -> dict:
    if "sleep_hours" in d and isinstance(d["sleep_hours"], str):
        try: d["sleep_hours"] = float(re.sub(r"[^\d\.]", "", d["sleep_hours"]))
        except: d["sleep_hours"] = None
    if "mood_1to5" in d and isinstance(d["mood_1to5"], str):
        try: d["mood_1to5"] = int(re.sub(r"[^\d]", "", d["mood_1to5"]))
        except: d["mood_1to5"] = None
    return {k:v for k,v in d.items() if v is not None}

def recommended_order_by_time(enabled_keys: list[str]) -> list[str]:
    hh = time.localtime().tm_hour
    base = ["sleep_hours","pain","dizzy","mood_1to5","med_taken","meal_morning"]
    if 5 <= hh < 11: base = ["meal_morning","sleep_hours","pain","dizzy","mood_1to5","med_taken"]
    if 11 <= hh < 17: base = ["sleep_hours","pain","dizzy","mood_1to5","med_taken","meal_morning"]
    return [k for k in base if k in enabled_keys]

def build_cards(enabled_keys: list[str]) -> list[dict]:
    return [{"id": TOPIC_REGISTRY[k]["id"], "key": k,
             "prompt": TOPIC_REGISTRY[k]["prompt"], "type": TOPIC_REGISTRY[k]["type"]}
            for k in enabled_keys]

def build_state_capsule(phase, current_card, signals, target):
    return (
        f"<STATE>\nphase: {phase}\n"
        f"current_card: {current_card['key'] if current_card else 'null'}\n"
        f"signals: {json.dumps(signals, ensure_ascii=False)}\n"
        f"target_signals: {target}\n</STATE>"
    )

# ========= I/O =========
def get_user_input_text():
    return input("You> ").strip()

def get_user_input_voice_streaming_space(speech_client, device_in=None, samplerate=16000):
    from pynput import keyboard; import sounddevice as sd
    pressed = threading.Event()
    def on_press(key):
        if key == keyboard.Key.space: pressed.set()
    def on_release(key):
        if key == keyboard.Key.space: pressed.clear(); return False

    print("⏺ Hold SPACE to talk; release to stop.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as _:
        while not pressed.is_set(): time.sleep(0.02)
        CHUNK_MS = 20; frames_per_block = int(samplerate*CHUNK_MS/1000); audio_q = queue.Queue()
        def cb(indata, frames, t, status): audio_q.put(bytes(indata))
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=samplerate, language_code="en-US",
            enable_automatic_punctuation=True, model="latest_short",
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True, single_utterance=False)
        final_text, last_print_len = "", 0
        def audio_request_gen():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            with sd.RawInputStream(samplerate=samplerate, blocksize=frames_per_block,
                                   dtype='int16', channels=1, callback=cb, device=device_in):
                while pressed.is_set() or not audio_q.empty():
                    try: buf = audio_q.get(timeout=0.05); yield speech.StreamingRecognizeRequest(audio_content=buf)
                    except queue.Empty: pass
        responses = speech_client.streaming_recognize(requests=audio_request_gen())
        try:
            for resp in responses:
                for result in resp.results:
                    if not result.alternatives: continue
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        erase = " " * max(0, last_print_len - len(transcript))
                        print("\rYou> " + transcript + erase)
                        final_text = transcript.strip(); last_print_len = 0
                    else:
                        line = "(listening) " + transcript; last_print_len = len(line)
                        print("\r" + line, end="", flush=True)
        except Exception as e:
            print(f"\n[stream error] {e}")
    return final_text.strip()

def speak_gcloud_tts(tts_client: tts.TextToSpeechClient, text: str, enabled: bool):
    if not enabled or not text: return
    try:
        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-A")
        sample_rate = 22050
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, speaking_rate=0.95, sample_rate_hertz=sample_rate)
        res = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        pcm = res.audio_content
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f: tmp = f.name
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(pcm)
        try:
            import platform, winsound
            if platform.system().lower().startswith("win"): winsound.PlaySound(tmp, winsound.SND_FILENAME); os.remove(tmp)
            else: print(f"(Saved audio to) -> {tmp}")
        except Exception:
            print(f"(Playback failed. Saved to) -> {tmp}")
    except Exception as e:
        print(f"(TTS error: {e})")

def make_gcloud_clients(embedded_json: str):
    if embedded_json and embedded_json.strip().startswith("{") and embedded_json.strip() != "{}":
        info = json.loads(embedded_json)
        return (speech.SpeechClient.from_service_account_info(info), tts.TextToSpeechClient.from_service_account_info(info))
    return speech.SpeechClient(), tts.TextToSpeechClient()

def time_greeting():
    hh = time.localtime().tm_hour
    if 5 <= hh < 11:  return "Good morning. How has your morning been?"
    if 11 <= hh < 17: return "Hello! How’s your day going?"
    return "Good evening. How was your day?"

# ========= Main (CHAT -> CHECK batch) =========
class Phase(Enum):
    CHAT = 1
    CHECK = 2

CHAT_TURNS_BEFORE_CHECK = 5
MAX_TURNS = 60
CHECK_ANNOUNCE = "Let’s do a quick health check for today. Just a few questions, it’ll be quick."
PRAISES = ["Thanks for sharing.", "That helps.", "Great, thank you.", "Got it, thanks."]

END_PAT = re.compile(
    r"(?:\b(end|stop|finish|bye|goodbye)\b|\bsee you\b|\bthat'?s it\b)",
    re.I
)

def is_end_intent_en(text: str) -> bool:
    s = (text or "").strip().lower()
    # optional: prefer when it's at the end of the utterance
    if END_PAT.search(s):
        return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Two-phase (CHAT -> CHECK batch) CLI [English]")
    ap.add_argument("--mode", choices=["TEXT", "VOICE"], default="TEXT")
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--device-in", default=None)
    ap.add_argument("--api-key", default=None, help="Gemini API key (overrides env)")
    ap.add_argument("--check-keys", default="sleep_hours,meal_morning,med_taken",
                    help="Comma-separated keys (e.g., sleep_hours,meal_morning,med_taken | all)")
    args = ap.parse_args()

    # Enabled keys
    if args.check_keys.strip().lower() == "all":
        enabled_keys = list(TOPIC_REGISTRY.keys())
    else:
        enabled_keys = [k.strip() for k in args.check_keys.split(",") if k.strip() in TOPIC_REGISTRY]
        if not enabled_keys:
            print("No valid --check-keys provided. Choose from TOPIC_REGISTRY keys.")
            sys.exit(1)

    ACTIVE_CARDS = build_cards(enabled_keys)
    TARGET_SIGNALS = len(ACTIVE_CARDS)
    ORDERED_KEYS = recommended_order_by_time(enabled_keys)

    # Dynamic CFGs
    EXTRACT_GENERAL_CFG = build_extract_general_cfg(enabled_keys)
    GUARD_CFG = build_guard_cfg(enabled_keys)

    # Gemini
  api_key = args.api_key or EMBEDDED_API_KEY
  if not api_key:
    print("Gemini API key not found. Pass --api-key or set GEMINI_API_KEY."); sys.exit(1)

  # Configure genai and create one model client object to reuse
  genai.configure(api_key=api_key)
  MODEL_CLIENT = genai.GenerativeModel(MODEL)


    # GCP
    try:
        speech_client, tts_client = make_gcloud_clients(EMBEDDED_GCLOUD_JSON)
    except Exception as e:
        print("Failed to init GCP clients:", e)
        print("Fix by embedding GCLOUD_SA_JSON or setting GOOGLE_APPLICATION_CREDENTIALS.")
        sys.exit(1)

    # Session
    SESSION = {
        "phase": Phase.CHAT,
        "turn": 0,
        "chat_turns": 0,
        "signals": {k: None for k in enabled_keys},
        "check_plan": [], "check_idx": 0,
        "awaiting_answer": False, "asked_once": False, "clarify_done": False,
    }

    def build_check_plan():
        plan = []
        for k in ORDERED_KEYS:
            if SESSION["signals"][k] is None:
                plan.append(next(c for c in ACTIVE_CARDS if c["key"] == k))
        return plan

    # History & kickoff
    history: list[types.Content] = []
    first = time_greeting()
    print("\nRobot> " + first + "\n"); speak_gcloud_tts(tts_client, first, enabled=args.speak)
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=first)]))
    SESSION["phase"] = Phase.CHAT; SESSION["chat_turns"] = 1

    try:
        while True:
            # Input
            if args.mode == "TEXT":
                u = get_user_input_text()
                if u.lower() in ("q", "quit", "exit"): print("Ending. Bye!"); break
                if not u: continue
            else:
                u = get_user_input_voice_streaming_space(speech_client, device_in=args.device_in)
                if not u: print(""); continue

            # End words
            if is_end_intent_en(u):
                print("That’s all for today. Thanks for chatting!")
                break

            # Extraction
            ex = {}
            if SESSION["phase"] == Phase.CHECK and SESSION["awaiting_answer"] and SESSION["check_plan"]:
                card = SESSION["check_plan"][SESSION["check_idx"]]
                ex = llm_extract_focused(g_client, card, u)
                if not ex and card["type"] == "number":
                    m = re.search(r"(\d+(?:\.\d+)?)", u)
                    if m:
                        key = card["key"]; val = float(m.group(1))
                        if key == "mood_1to5":
                            val = min(5, max(1, int(round(val))))
                        ex = {key: val}
            if not ex:
                ex = llm_extract_general(g_client, u, EXTRACT_GENERAL_CFG)
            for k, v in (ex or {}).items():
                if k in SESSION["signals"] and v is not None:
                    SESSION["signals"][k] = v

            # Turn count
            SESSION["turn"] += 1

            # CHAT -> CHECK transition
            if SESSION["phase"] == Phase.CHAT:
                SESSION["chat_turns"] += 1
                if SESSION["chat_turns"] >= CHAT_TURNS_BEFORE_CHECK:
                    announce = CHECK_ANNOUNCE
                    SESSION["check_plan"] = build_check_plan()
                    if not SESSION["check_plan"]:
                        print("\nRobot> " + announce + "\n"); speak_gcloud_tts(tts_client, announce, enabled=args.speak)
                        print("(That’s all for today. Thanks for chatting!)"); break
                    first_card = SESSION["check_plan"][0]
                    msg = f"{announce}\n{random.choice(PRAISES)} {first_card['prompt']}"
                    print("\nRobot> " + msg + "\n"); speak_gcloud_tts(tts_client, msg, enabled=args.speak)
                    history.append(types.Content(role="model", parts=[types.Part.from_text(text=msg)]))
                    SESSION["phase"] = Phase.CHECK
                    SESSION["check_idx"] = 0
                    SESSION["awaiting_answer"] = True
                    SESSION["asked_once"] = True
                    SESSION["clarify_done"] = False
                    continue

            # Generate / progress
            user_c = types.Content(role="user", parts=[types.Part.from_text(text=u)])
            history.append(user_c)

            if SESSION["phase"] == Phase.CHAT:
                state_capsule = build_state_capsule("CHAT", None, SESSION["signals"], TARGET_SIGNALS)
                try:
                    text = gen_chat_with_guard(g_client, history[:-1], u, state_capsule, GUARD_CFG)
                except exceptions.GoogleAPICallError as e:
                    print("API call error:", e)
                except exceptions.RetryError as e:
                    print("Retry error:", e)
                except Exception as e:
                    print("Unexpected error:", e)
                if not text: text = "Thanks."
                if len(text) > 400: text = text[:380] + "... (let me know if you want more)"
                print("\nRobot> " + text + "\n"); speak_gcloud_tts(tts_client, text, enabled=args.speak)
                history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
            else:
                # CHECK flow
                def advance_to_next_card():
                    SESSION["check_idx"] += 1
                    SESSION["asked_once"] = False
                    SESSION["clarify_done"] = False
                    SESSION["awaiting_answer"] = False

                # Skip any already-filled signals
                while SESSION["check_plan"] and SESSION["signals"].get(SESSION["check_plan"][SESSION["check_idx"]]["key"]) is not None:
                    advance_to_next_card()
                    if SESSION["check_idx"] >= len(SESSION["check_plan"]): break

                if SESSION["check_idx"] >= len(SESSION["check_plan"]):
                    print("(That’s all for today. Thanks for chatting!)")
                    break

                card = SESSION["check_plan"][SESSION["check_idx"]]
                key = card["key"]
                text = None

                if SESSION["asked_once"] and SESSION["awaiting_answer"]:
                    if SESSION["signals"].get(key) is None and not SESSION["clarify_done"]:
                        # One clarification
                        if card["type"] == "bool":
                            text = random.choice(PRAISES) + " Please answer in one or two words."
                        elif card["id"] == "sleep":
                            text = random.choice(PRAISES) + " A number please. About how many hours did you sleep?"
                        elif card["id"] == "mood":
                            text = random.choice(PRAISES) + " On a 1–5 scale, how is it?"
                        else:
                            text = random.choice(PRAISES) + " " + card["prompt"]
                        SESSION["clarify_done"] = True
                    elif SESSION["signals"].get(key) is None and SESSION["clarify_done"]:
                        # Give up this card; move on
                        advance_to_next_card()
                        if SESSION["check_idx"] < len(SESSION["check_plan"]):
                            next_card = SESSION["check_plan"][SESSION["check_idx"]]
                            text = f"{random.choice(PRAISES)} {next_card['prompt']}"
                            SESSION["asked_once"] = True; SESSION["awaiting_answer"] = True
                        else:
                            print("(That’s all for today. Thanks for chatting!)"); break
                    else:
                        # We got it; move on
                        advance_to_next_card()
                        if SESSION["check_idx"] < len(SESSION["check_plan"]):
                            next_card = SESSION["check_plan"][SESSION["check_idx"]]
                            text = f"{random.choice(PRAISES)} {next_card['prompt']}"
                            SESSION["asked_once"] = True; SESSION["awaiting_answer"] = True
                        else:
                            print("(That’s all for today. Thanks for chatting!)"); break
                else:
                    # Safety: usually not hit
                    text = f"{random.choice(PRAISES)} {card['prompt']}"
                    SESSION["asked_once"] = True; SESSION["awaiting_answer"] = True

                if not text: text = "Thanks."
                if len(text) > 400: text = text[:380] + "... (let me know if you want more)"
                print("\nRobot> " + text + "\n"); speak_gcloud_tts(tts_client, text, enabled=args.speak)
                history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))

            # Safety end
            if SESSION["turn"] >= MAX_TURNS:
                print("(That’s all for today. Thanks for chatting!)"); break

    except KeyboardInterrupt:
        print("\nEnd")

if __name__ == "__main__":
    main()
