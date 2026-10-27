"""Build compact video contact sheets for manual PushT failure review."""

from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "results" / "pusht_videos"
FRAME_SIZE = 190
FRAME_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.99)


def make_sheet(group: str, first_episode: int) -> Path:
    episodes = range(first_episode, first_episode + 5)
    margin = 8
    label_height = 25
    width = margin * 2 + FRAME_SIZE * len(FRAME_FRACTIONS)
    height = margin * 2 + (FRAME_SIZE + label_height) * len(episodes)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for row, episode in enumerate(episodes):
        video_path = VIDEOS / group / f"eval_episode_{episode}.mp4"
        reader = imageio.get_reader(video_path)
        try:
            n_frames = reader.count_frames()
            y = margin + row * (FRAME_SIZE + label_height)
            draw.text((margin + 4, y), f"{group}: seed {1000 + episode} / episode {episode}", fill="black")
            for col, fraction in enumerate(FRAME_FRACTIONS):
                index = min(round((n_frames - 1) * fraction), n_frames - 1)
                frame = Image.fromarray(reader.get_data(index)).resize((FRAME_SIZE, FRAME_SIZE))
                sheet.paste(frame, (margin + col * FRAME_SIZE, y + label_height))
        finally:
            reader.close()
    output = ROOT / "results" / f"pusht_contact_{group}_{first_episode:02d}-{first_episode + 4:02d}.png"
    sheet.save(output)
    return output


def main() -> None:
    for group in ("act100", "act10", "diffusion"):
        for first in (0, 5):
            print(make_sheet(group, first))


if __name__ == "__main__":
    main()
