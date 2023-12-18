import streamlit as st
from io import BytesIO
import subprocess
import torchaudio
import torch
from demucs import pretrained
from demucs.apply import apply_model
import time

st.title("mp3パート分解アプリ")

st.write("5~10分くらい時間かかるよ！")
st.write("mp3と同じ場所に「ファイル名_drums.mp3」「ファイル名_bass.mp3」「ファイル名_vocals.mp3」「ファイル名_others.mp3」の４つのファイルが出力されるよ！")
# uploadfile_path = st.file_uploader("1. ローカルのmp3ファイルを指定してね！")
uploadfile_path = st.text_input('ローカルにある分解したいmp3ファイルのパスを貼り付けてね！')

if st.button("パス貼り付けたらここを押してね！"):
    button_pressed_time = time.time()  # ボタンを押した時刻を取得
    
    filename = uploadfile_path.split("/")[-1].split(".")[0]
    filename_path = uploadfile_path.split(filename)[0]

    # Load the pretrained model
    model = pretrained.get_model('mdx')

    # Load the sample.mp3 audio file
    audio_file = uploadfile_path 
    waveform, sample_rate = torchaudio.load(audio_file, normalize=True)

    # Ensure stereo audio (2 channels)
    if waveform.shape[0] == 1:
        waveform = torch.cat([waveform, waveform], dim=0)

    # Apply the model to the audio data
    output_waveforms = apply_model(model, waveform.unsqueeze(0))[0]

    # Convert and save separated audio sources directly to MP3
    for i, source_waveform in enumerate(output_waveforms):
        # Convert WAV to MP3 using ffmpeg with adjusted options
        if i == 0:
            output_filename = filename_path + "/" + filename + "_drums.mp3"
        elif i == 1:
            output_filename = filename_path + "/" + filename + "_bass.mp3"
        elif i == 2:
            output_filename = filename_path + "/" + filename + "_others.mp3"
        elif i == 3:
            output_filename = filename_path + "/" + filename + "_vocals.mp3"
            
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", "2",
            "-i", "-",
            "-codec:a", "libmp3lame",
            "-b:a", "256k",  # Adjust the bitrate if needed
            output_filename
        ]

        # Create a BytesIO object to hold WAV data
        wav_buffer = BytesIO()

        # Save WAV data to BytesIO with format specified
        torchaudio.save(wav_buffer, source_waveform, sample_rate, format="wav")

        # Reset the buffer position to the beginning
        wav_buffer.seek(0)

        try:
            # Run ffmpeg command and pass the WAV data directly
            result = subprocess.run(ffmpeg_cmd, input=wav_buffer.read(), capture_output=True, check=True)
            print(f"ffmpeg stdout for {output_filename}:", result.stdout.decode())
            print(f"ffmpeg stderr for {output_filename}:", result.stderr.decode())
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error for {output_filename}:", e.stderr.decode())
     
                
    end_time = time.time()  # 処理が終了した時刻を取得
    elapsed_time = end_time - button_pressed_time
    elapsed_minutes = elapsed_time / 60 
    # 完了メッセージ
    st.success('完了!')
    st.write(f"かかった時間は約 {elapsed_minutes:.2f} 分だったよ！")  

