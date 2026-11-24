
import pygame
import numpy as np
import time

pygame.mixer.init()

frequency = 1000
duration = 500
sample_rate = 44100

samples = np.arange(int(sample_rate * duration / 1000))

wave = 0.5 * np.sin(
    2 * np.pi * frequency * samples / sample_rate
)

audio = (wave * 32767).astype(np.int16)

sound = pygame.mixer.Sound(buffer=audio.tobytes())

print("Playing test alert...")

sound.play()

time.sleep(1)

pygame.mixer.quit()
