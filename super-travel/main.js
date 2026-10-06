// Reveal-up: fade/slide content blocks in once 15% of them is on screen.
(function () {
  var items = document.querySelectorAll('.reveal');

  function show(el) {
    el.classList.add('is-visible');
    // Drop the stagger delay once the entrance has played so hover
    // transitions on the same element (e.g. service cards) stay instant.
    var delay = parseFloat(getComputedStyle(el).transitionDelay) * 1000 || 0;
    setTimeout(function () { el.style.setProperty('--d', '0ms'); }, delay + 1100);
  }

  if (!('IntersectionObserver' in window)) {
    items.forEach(show);
    return;
  }

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      show(entry.target);
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.15 });

  items.forEach(function (el) { observer.observe(el); });
})();

// Point every "plan a trip" link at the Streamlit concierge (see <meta name="chat-url">).
(function () {
  var meta = document.querySelector('meta[name="chat-url"]');
  if (!meta || !meta.content) return;
  // A link with data-chat-journey="<slug>" opens the chat already started on that journey.
  document.querySelectorAll('[data-chat-link]').forEach(function (a) {
    var slug = a.getAttribute('data-chat-journey');
    a.href = slug ? meta.content + (meta.content.indexOf('?') === -1 ? '?' : '&') + 'journey=' + encodeURIComponent(slug) : meta.content;
  });
})();
