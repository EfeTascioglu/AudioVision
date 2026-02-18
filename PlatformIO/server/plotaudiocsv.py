import plotly.express as px

with open(r"C:\Users\sparq\Videos\PlatformIO\AudioVision\server\audio_2026-02-14T19-22-26-813Z.csv", "r") as f:
    lines = f.readlines()

left = []
right = []
mono = []

def split_commas_toint(line):
    return [int(x.strip()) for x in line.strip().split(",")]

for i in range(1, len(lines), 3):
    try:
        left.append(split_commas_toint(lines[i])[1])
        right.append(split_commas_toint(lines[i + 1])[1])
        mono.append(split_commas_toint(lines[i + 2])[1])
    except (ValueError, IndexError) as e:
        print(f"Error at lines {i}-{i+2}: {e}, skipping these lines")
    
fig = px.line(x=list(range(len(left))), y=[left, right, mono], labels={"x": "Sample Index", "y": "Amplitude"}, title="Audio Samples")
fig.update_layout(legend_title_text="Channels", legend=dict(itemsizing="constant"))
fig.show()
