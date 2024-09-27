from espnet2.bin.tts_inference import Text2Speech

model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

speech, *_ = model("text to generate speech from")