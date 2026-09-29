// Highlight the current section in the sidebar and add copy buttons to code blocks.
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc li a'));
  var sections = links.map(function (a) { return document.querySelector(a.getAttribute('href')); }).filter(Boolean);
  if ('IntersectionObserver' in window && sections.length) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (l) { l.classList.toggle('active', l.getAttribute('href') === '#' + e.target.id); });
        }
      });
    }, { rootMargin: '-15% 0px -75% 0px' });
    sections.forEach(function (s) { io.observe(s); });
  }

  document.querySelectorAll('pre:not(.output)').forEach(function (pre) {
    var btn = document.createElement('button');
    btn.className = 'copy'; btn.type = 'button'; btn.textContent = 'Copy';
    btn.addEventListener('click', function () {
      var text = pre.innerText.replace(/^Copy\n?/, '').replace(/Copied!\n?$/, '');
      text = pre.querySelector('code') ? pre.querySelector('code').innerText : text;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = 'Copied!'; setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
        });
      }
    });
    pre.appendChild(btn);
  });
})();
