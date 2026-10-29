# Publishing the article on Medium

The article is `medium/article.md`. Its images are in `medium/images/`, numbered in the order they appear.

## 1. Preview it

```powershell
.venv\Scripts\python src\medium.py serve
```

Open http://127.0.0.1:8000/preview.html. The page is laid out like a Medium story, so you can read it as your readers will.

## 2. Before you copy

- Replace `[REPO_URL]` in the last section with your GitHub link (it is highlighted in the preview), or delete the sentence. Then rebuild with `src\medium.py build`.
- Optional: if you want the code on GitHub, push this project folder to a public repository first.

## 3. Copy into Medium

1. Click **Copy article** at the top of the preview page.
2. In Medium, choose **Write**, click in the empty story and paste (Ctrl+V).
   - The first line becomes the title and the grey line under it the subtitle.
   - Headings, bold text, lists, links and code blocks paste as they are.
3. Each image arrives as a line like
   `[UPLOAD IMAGE 03_annual_production.png | caption: ...]`.
   For each one:
   1. Delete the line.
   2. Click the **+** on the empty line, choose the image button and upload that file from `medium/images/`.
   3. Type the caption under the image (copy it from the marker line).
4. For each code block, check Medium kept it as a code block. If not, select the text and press Ctrl+Alt+6. You can also set the language to Python from the block's menu.

## 4. Publish settings

- **Story preview image:** `01_hero_heatmap.png`. Medium uses the first image by default.
- **Topics (maximum 5):** Data Science, Machine Learning, Time Series Forecasting, Python, India.
- **Story preview title and subtitle:** keep the article's title and subtitle.

Medium shows your name and profile as the author, so the article has no byline.

## Why images are uploaded by hand

Medium copies pasted images from their web address. The preview runs on your own computer, so Medium cannot reach those addresses, and each image needs one manual upload.
