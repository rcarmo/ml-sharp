"""Minimal web app for generating a panning video from a single image.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from .sharp_pan import generate_sharp_swipe_mp4


HTML_INDEX = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>SHARP — Swipe Trajectory</title>
  </head>
  <body>
    <main>
      <h1>SHARP swipe trajectory video</h1>
      <form action=\"/pan\" method=\"post\" enctype=\"multipart/form-data\">
        <p>
          <label>Image: <input type=\"file\" name=\"image\" accept=\"image/*\" required /></label>
        </p>
        <p>
          <label>Duration (s): <input type=\"number\" name=\"duration_s\" value=\"4.0\" step=\"0.1\" min=\"0.1\" /></label>
        </p>
        <p>
          <label>FPS: <input type=\"number\" name=\"fps\" value=\"30\" step=\"1\" min=\"1\" max=\"60\" /></label>
        </p>
        <p>
          <label>Motion:</label>
          <select name=\"motion_scale\">
            <option value=\"0.10\">Very small (10%)</option>
            <option value=\"0.20\" selected>Small (20%)</option>
            <option value=\"0.30\">Medium (30%)</option>
            <option value=\"0.50\">Large (50%)</option>
            <option value=\"1.00\">Full (100%)</option>
          </select>
        </p>
        <p>
          <label>Wobble:</label>
          <select name=\"wobble_scale\">
            <option value=\"0.00\">Off</option>
            <option value=\"0.25\" selected>Subtle</option>
            <option value=\"0.50\">Medium</option>
            <option value=\"1.00\">More</option>
          </select>
        </p>
        <button type=\"submit\">Generate MP4</button>
      </form>
      <p>Trajectory: centered horizontal swipe with a subtle vertical cosine wobble; “Motion” controls the travel width.</p>
    </main>
  </body>
</html>
"""


def create_app():
    """Create the FastAPI app.

    Split into a factory function so the module can be imported even when
    FastAPI isn't installed.
    """

    try:
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import HTMLResponse, Response
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is not installed. Install web deps with: "
            "pip install -r requirements-web.txt"
        ) from exc

    app = FastAPI(title="sharp-pan")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_INDEX

    @app.post("/pan")
    async def pan(
        image: UploadFile = File(...),
        duration_s: float = Form(4.0),
        fps: int = Form(30),
        motion_scale: float = Form(0.20),
      wobble_scale: float = Form(0.25),
    ):
        image_bytes = await image.read()
        video_bytes = generate_sharp_swipe_mp4(
            image_bytes,
            duration_s=duration_s,
            fps=fps,
            motion_scale=motion_scale,
        wobble_scale=wobble_scale,
        )

        headers = {"Content-Disposition": "attachment; filename=pan.mp4"}
        return Response(content=video_bytes, media_type="video/mp4", headers=headers)

    return app


# Convenience for `uvicorn sharp.web.app:app`
app = create_app()
