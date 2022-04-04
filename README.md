# Audio2Tones

Convert an audiofile to tones for arduino or similar SoCs

## Installation
```
git clone https://github.com/Jacotsu/audio2tone.git
cd audio2tone
pip3 install .
```

## Usage
```
audio2tones.py {filename}
```

The resulting tones array will be written in `result.txt` in the following format
`[duration in microseconds, tone frequency bin selector, tone volume]`
