import sys
import subprocess
import os
import mutagen
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TCON

def get_tags_and_add(dir, verbose=1):
    for file in os.listdir(dir):
        if verbose > 0:
            print(f'sending {file} to API')
        sub = subprocess.check_output([f'curl -F "audio=@Mp3ToPCM/{file};type=audio/wav" -XPOST http://localhost:5000/model/predict'],  shell = True)
        response_dict = eval(sub)
        tags_to_add = []
        try:
            audio = ID3(f'Mp3/{file[0:-3]}mp3')
        except:
            audio = MP3(f'Mp3/{file[0:-3]}mp3')
            audio.add_tags()
            audio.save()
            audio = ID3(f'Mp3/{file[0:-3]}mp3')
        for pred in response_dict['predictions']:
            if pred['probability']>.25:
                tags_to_add.append(pred['label'])
        if verbose > 0:
            print('Adding these tags: ' + ', '.join(tags_to_add))
            print('-------------------------------')
        existing = audio.get('TCON')
        if existing:
            audio.add(TCON(text=' '.join(set(audio.get('TCON').text[0].split() + ['Beatboxing', 'Computer', 'keyboard']))))
        else:
            audio.add(TCON(text=' '.join(tags_to_add)))
        audio.save()

if __name__ == "__main__":
    get_tags_and_add(sys.argv[1])

