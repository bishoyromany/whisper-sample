import whisper
import Levenshtein

model = whisper.load_model("base.en")

result = model.transcribe('stand-up.mp3', fp16 = False, language="en")

print(f"Text: {result['text']}\n")
print(f"Language: {result['language']}")

rate = Levenshtein.ratio('Stand up.', result['text'])

print(f"Score Rate: {rate}")