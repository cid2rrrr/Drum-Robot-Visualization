import pyaudio, wave


# 녹음 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050  # 샘플링 레이트 (Hz)
CHUNK = 512 * 32 # 버퍼 크기 (프레임 수)


# 녹음된 오디오를 저장할 WAV 파일 설정
OUTPUT_FILENAME = "output.wav"




if __name__ == "__main__":
    audio_frames = []

    # PyAudio 초기화
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=3)

    try:
        while True:
            audio_data = stream.read(CHUNK)
            audio_frames.append(audio_data)

            if len(audio_frames) > 0:
                output_wave = wave.open(OUTPUT_FILENAME, 'wb')
                output_wave.setnchannels(CHANNELS)
                output_wave.setsampwidth(2)
                output_wave.setframerate(RATE)
                output_wave.writeframes(b''.join(audio_frames))
                output_wave.close()                

    except KeyboardInterrupt:
        pass

    # 스트림 정리 및 WAV 파일 닫기
    stream.stop_stream()
    stream.close()
    audio.terminate()
    