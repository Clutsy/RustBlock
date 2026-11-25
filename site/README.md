# RustBlock showcase

A static, English-only landing page for RustBlock. No build step and no runtime
dependencies: Vercel serves `index.html`, `styles.css`, `script.js` and the
`assets/` folder directly.

## Preview locally

From the repository root:

```bash
cd site
python -m http.server 4173
```

Open <http://127.0.0.1:4173>.

## Deploy to Vercel

1. Import <https://github.com/Clutsy/RustBlock> in the Vercel dashboard.
2. Set **Root Directory** to `site`.
3. Framework preset: **Other**. Build command: empty. Output directory: `.`.
4. Deploy.

Or with the CLI:

```bash
npx vercel --cwd site --prod
```

`site/vercel.json` adds clean URLs, long-lived asset caching and basic security
headers. The download buttons read the latest GitHub Release at runtime and fall
back to the releases page when the API is unavailable.
