(() => {
  const RELEASES = "https://github.com/Clutsy/RustBlock/releases/latest";
  const API = "https://api.github.com/repos/Clutsy/RustBlock/releases/latest";

  const setText = (selector, value) => {
    document.querySelectorAll(selector).forEach((node) => {
      node.textContent = value;
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
      const assets = Array.isArray(release.assets) ? release.assets : [];
      const installer = assets.find((asset) =>
        /^RustBlock-Setup-.*\.exe$/i.test(asset.name || "")
      );
      const portable = assets.find((asset) =>
        /^RustBlock-.*-portable\.zip$/i.test(asset.name || "")
      );

      if (version) {
        setText("[data-version-label]", `v${version}`);
        document.title = `RustBlock ${version} — native GGUF conversion`;
      }
      if (installer) {
        document
          .querySelectorAll('[data-download="installer"]')
          .forEach((link) => (link.href = installer.browser_download_url));
      }
      if (portable) {
        document
          .querySelectorAll('[data-download="portable"]')
          .forEach((link) => (link.href = portable.browser_download_url));
      }
    } catch {
      document.querySelectorAll('[data-download]').forEach((link) => {
        if (!link.href || link.href === window.location.href) link.href = RELEASES;
      });
    }
  }

  const header = document.querySelector("[data-header]");
  const onScroll = () => {
    if (header) header.classList.toggle("is-scrolled", window.scrollY > 12);
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  const toggle = document.querySelector("[data-nav-toggle]");
  const nav = document.querySelector("[data-nav]");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("a").forEach((link) =>
      link.addEventListener("click", () => {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      })
    );
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const revealTargets = document.querySelectorAll(
    ".feature-card, .download-card, .steps li, .faq details, .split-copy, .split-visual, .discover-copy, .discover-frame"
  );
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
      { threshold: 0.12 }
    );
    revealTargets.forEach((node) => {
      node.classList.add("reveal");
      observer.observe(node);
    });
  }

  const year = document.querySelector("[data-year]");
  if (year) year.textContent = String(new Date().getFullYear());

  loadRelease();
})();
