(() => {
  document.documentElement.classList.add("js");

  const REPO = "Clutsy/RustBlock";
  const RELEASES = `https://github.com/${REPO}/releases`;
  const API = `https://api.github.com/repos/${REPO}/releases/latest`;
  const REPO_API = `https://api.github.com/repos/${REPO}`;
  const ALL_RELEASES_API = `https://api.github.com/repos/${REPO}/releases?per_page=100`;
  const COMMUNITY_CACHE = "rustblock:github:v1";
  const COMMUNITY_TTL = 10 * 60 * 1000;
  const number = new Intl.NumberFormat("en-US");

  const formatSize = (bytes) => {
    if (!Number.isFinite(bytes) || bytes <= 0) return "";
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const setText = (selector, value) => {
    if (!value) return;
    document.querySelectorAll(selector).forEach((node) => {
      node.textContent = value;
    });
  };

  const setLink = (selector, href) => {
    if (!href) return;
    document.querySelectorAll(selector).forEach((node) => {
      node.href = href;
    });
  };

  const findAsset = (assets, pattern) =>
    assets.find((asset) => pattern.test(asset.name || ""));

  const bindAsset = (selector, asset, sizeKey) => {
    if (!asset) return;
    setLink(selector, asset.browser_download_url);
    if (asset.size) setText(`[data-size="${sizeKey}"]`, formatSize(asset.size));
  };

  async function loadRelease() {
    try {
      const response = await fetch(API, {
        headers: { Accept: "application/vnd.github+json" },
      });
      if (!response.ok) throw new Error(`GitHub ${response.status}`);
      const release = await response.json();
      const version = String(release.tag_name || "").replace(/^v/i, "");
      const label = version ? `v${version}` : "";
      const assets = Array.isArray(release.assets) ? release.assets : [];
      const installer = findAsset(assets, /^RustBlock-Setup-.*\.exe$/i);
      const portable = findAsset(assets, /^RustBlock-.*-portable\.zip$/i);
      const linux = findAsset(assets, /^RustBlock-.*-linux-x86_64\.tar\.gz$/i);
      const macArm = findAsset(assets, /^RustBlock-.*-macos-arm64\.tar\.gz$/i);
      const macIntel = findAsset(assets, /^RustBlock-.*-macos-x86_64\.tar\.gz$/i);
      const checksums = findAsset(assets, /^SHA256SUMS\.txt$/i);

      if (label) {
        document.querySelectorAll("[data-version]").forEach((node) => {
          if (node.hasAttribute("data-size")) {
            const size = node.textContent.split("·")[0].trim();
            node.textContent = `${size} · ${label}`;
          } else {
            node.textContent = label;
          }
        });
        document.title = `RustBlock ${version} — native GGUF conversion`;
      }

      if (installer) {
        setLink("[data-installer]", installer.browser_download_url);
        if (installer.size) {
          document.querySelectorAll('[data-size="installer"]').forEach((node) => {
            const suffix = node.hasAttribute("data-version") && label ? ` · ${label}` : "";
            node.textContent = `${formatSize(installer.size)}${suffix}`;
          });
        }
      }

      bindAsset("[data-portable]", portable, "portable");
      bindAsset("[data-linux]", linux, "linux");
      bindAsset("[data-macos-arm64]", macArm, "macos-arm64");
      bindAsset("[data-macos-x64]", macIntel, "macos-x64");

      if (checksums) setLink("[data-checksums]", checksums.browser_download_url);
      setLink("[data-release]", release.html_url);
      setLink("[data-releases]", RELEASES);
    } catch {
      const latest = `${RELEASES}/latest`;
      ["[data-installer]", "[data-portable]", "[data-linux]", "[data-macos-arm64]", "[data-macos-x64]"].forEach(
        (selector) => setLink(selector, latest)
      );
    }
  }

  const readCommunityCache = () => {
    try {
      const raw = sessionStorage.getItem(COMMUNITY_CACHE);
      if (!raw) return null;
      const cached = JSON.parse(raw);
      if (!cached || typeof cached.ts !== "number" || !cached.data) return null;
      return cached;
    } catch {
      return null;
    }
  };

  const renderCommunity = (data) => {
    document.querySelectorAll("[data-stars]").forEach((node) => {
      node.textContent = number.format(data.stars || 0);
    });
    document.querySelectorAll("[data-forks]").forEach((node) => {
      node.textContent = number.format(data.forks || 0);
    });
    document.querySelectorAll("[data-downloads]").forEach((node) => {
      node.textContent = number.format(data.downloads || 0);
    });
    if (data.latestTag) {
      document.querySelectorAll("[data-latest]").forEach((node) => {
        node.textContent = data.latestTag;
      });
    }
    if (data.latestUrl) setLink("[data-latest]", data.latestUrl);
  };

  async function loadCommunity() {
    const cached = readCommunityCache();
    if (cached) {
      renderCommunity(cached.data);
      if (Date.now() - cached.ts < COMMUNITY_TTL) return;
    }

    try {
      const [repoResponse, releasesResponse] = await Promise.all([
        fetch(REPO_API, { headers: { Accept: "application/vnd.github+json" } }),
        fetch(ALL_RELEASES_API, { headers: { Accept: "application/vnd.github+json" } }),
      ]);
      if (!repoResponse.ok || !releasesResponse.ok) {
        throw new Error("GitHub API unavailable");
      }
      const repo = await repoResponse.json();
      const releases = await releasesResponse.json();
      const list = Array.isArray(releases) ? releases : [];
      const downloads = list.reduce(
        (total, release) =>
          total +
          (Array.isArray(release.assets) ? release.assets : [])
            .filter((asset) => !/^SHA256SUMS\.txt$/i.test(asset.name || ""))
            .reduce((sum, asset) => sum + (Number(asset.download_count) || 0), 0),
        0
      );
      const latest = list.find((release) => !release.draft && !release.prerelease) || list[0];
      const data = {
        stars: Number(repo.stargazers_count) || 0,
        forks: Number(repo.forks_count) || 0,
        downloads,
        latestTag: latest?.tag_name || "",
        latestUrl: latest?.html_url || `${RELEASES}/latest`,
      };
      renderCommunity(data);
      try {
        sessionStorage.setItem(COMMUNITY_CACHE, JSON.stringify({ ts: Date.now(), data }));
      } catch {
        /* private mode */
      }
    } catch {
      /* keep the cached values or the em dashes */
    }
  }

  const header = document.querySelector("[data-header]");
  const onScroll = () => {
    if (header) header.classList.toggle("is-scrolled", window.scrollY > 8);
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  const toggle = document.querySelector("[data-nav-toggle]");
  const nav = document.querySelector("[data-nav]");
  const closeNav = () => {
    if (!nav || !toggle) return;
    nav.classList.remove("is-open");
    toggle.setAttribute("aria-expanded", "false");
  };

  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", closeNav);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeNav();
    });
    window.addEventListener("resize", () => {
      if (window.innerWidth > 920) closeNav();
    });
  }

  const faq = document.querySelectorAll(".faq details");
  faq.forEach((item) => {
    item.addEventListener("toggle", () => {
      if (!item.open) return;
      faq.forEach((other) => {
        if (other !== item) other.open = false;
      });
    });
  });

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const targets = document.querySelectorAll("[data-reveal]");
  if (!reduceMotion && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    targets.forEach((node) => observer.observe(node));
  } else {
    targets.forEach((node) => node.classList.add("in"));
  }

  const year = document.querySelector("[data-year]");
  if (year) year.textContent = String(new Date().getFullYear());

  loadRelease();
  loadCommunity();
})();
