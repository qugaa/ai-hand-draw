"""Generate the application icon.

    python assets/make_icon.py

Draws at high resolution and downsamples into a multi-size .ico so the icon
stays crisp everywhere Windows shows it (16px tray -> 256px file view).

The mark: a fingertip trailing a drawn stroke - the app's whole idea, and it
still reads at 16 pixels where a detailed hand would turn to mush.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024  # final working resolution
SS = 3  # supersampling factor; edges are drawn big and shrunk to antialias
OUT = Path(__file__).with_name("icon.ico")

BG_TOP = (99, 91, 255)      # indigo
BG_BOTTOM = (176, 66, 243)  # violet
INK = (255, 255, 255)


def _bezier(p0, p1, p2, p3, steps= 220):
    """Cubic bezier as a list of points."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _stamp_stroke(target, points, color, w_start, w_end):
    """Draw a stroke by stamping overlapping discs along ``points``.

    PIL's wide polylines leave serrated edges where segments join; stamping
    discs gives a clean outline and lets the stroke taper like a pen line.
    """
    draw = ImageDraw.Draw(target)
    last = len(points) - 1
    for i, (x, y) in enumerate(points):
        r = (w_start + (w_end - w_start) * (i / last)) / 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def _rounded_gradient(size: int) -> Image.Image:
    """Vertical gradient clipped to a rounded square."""
    grad = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / (size - 1)
        grad.putpixel(
            (0, y),
            tuple(int(BG_TOP[c] + (BG_BOTTOM[c] - BG_TOP[c]) * t) for c in range(3)),
        )
    grad = grad.resize((size, size))

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=int(size * 0.22), fill=255
    )

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(grad, (0, 0), mask)
    return canvas


def build() -> Path:
    s = SIZE * SS
    img = _rounded_gradient(s)

    # The stroke the fingertip has just drawn: one confident upward sweep,
    # rising left-to-right so the mark reads as motion rather than a squiggle.
    curve = _bezier(
        (s * 0.17, s * 0.760),
        (s * 0.36, s * 0.815),
        (s * 0.47, s * 0.430),
        (s * 0.755, s * 0.335),
        steps=1600,
    )

    # Thin where the stroke started, full width under the fingertip.
    w_start, w_end = s * 0.030, s * 0.070

    # Soft shadow under the stroke so it lifts off the gradient.
    shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    _stamp_stroke(shadow, curve, (38, 0, 78, 120), w_start, w_end)
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(s * 0.016)))

    _stamp_stroke(img, curve, INK + (255,), w_start, w_end)
    draw = ImageDraw.Draw(img)

    # The fingertip leading the stroke: a bright dot with a halo ring.
    hx, hy = curve[-1]
    halo_r = int(s * 0.132)
    ring = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse(
        (hx - halo_r, hy - halo_r, hx + halo_r, hy + halo_r),
        outline=INK + (140,),
        width=int(s * 0.019),
    )
    img.alpha_composite(ring)

    dot_r = int(s * 0.078)
    draw.ellipse((hx - dot_r, hy - dot_r, hx + dot_r, hy + dot_r), fill=INK + (255,))

    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    sizes = [(n, n) for n in (256, 128, 64, 48, 32, 16)]
    img.save(OUT, format="ICO", sizes=sizes)
    img.resize((512, 512), Image.LANCZOS).save(OUT.with_name("icon.png"))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
