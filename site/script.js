(() => {
  document.documentElement.classList.add("js");

  const REPO = "Clutsy/RustBlock";
  const RELEASES = `https://github.com/${REPO}/releases`;
  const API = `https://api.github.com/repos/${REPO}/releases/latest`;

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
      const find = (pattern) =>
        assets.find((asset) => pattern.test(asset.name || ""));
      const installer = find(/^RustBlock-Setup-.*\.exe$/i);
      const portable = find(/^RustBlock-.*-portable\.zip$/i);
      const checksums = find(/^SHA256SUMS\.txt$/i);

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

      if (portable) {
        setLink("[data-portable]", portable.browser_download_url);
        if (portable.size) {
          setText('[data-size="portable"]', formatSize(portable.size));
        }
      }

      if (checksums) setLink("[data-checksums]", checksums.browser_download_url);
      setLink("[data-release]", release.html_url);
      setLink("[data-releases]", RELEASES);
    } catch {
      document.querySelectorAll("[data-installer]").forEach((node) => {
        node.href = `${RELEASES}/latest`;
      });
      document.querySelectorAll("[data-portable]").forEach((node) => {
        node.href = `${RELEASES}/latest`;
      });
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
})();
