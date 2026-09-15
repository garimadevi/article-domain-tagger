// Smooth scroll + navbar highlight + fade-in
document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
  anchor.addEventListener('click', function (e) {
    var target = document.querySelector(this.getAttribute('href'));
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  });
});
window.addEventListener('scroll', function () {
  var navbar = document.querySelector('.navbar');
  if (navbar) navbar.style.boxShadow = window.scrollY > 50 ? '0 4px 6px -1px rgba(0,0,0,0.1)' : 'none';
  var current = '';
  document.querySelectorAll('main section[id], header[id]').forEach(function (section) {
    if (window.scrollY >= section.offsetTop - 120) current = section.getAttribute('id');
  });
  document.querySelectorAll('.nav-links a').forEach(function (link) {
    link.classList.toggle('active', link.getAttribute('href') === '#' + current);
  });
});
(function () {
  var els = document.querySelectorAll('.section > .container > *');
  if (!('IntersectionObserver' in window)) return;
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) { en.target.style.opacity = '1'; en.target.style.transform = 'translateY(0)'; obs.unobserve(en.target); }
    });
  }, { threshold: 0.08 });
  els.forEach(function (el) {
    el.style.opacity = '0'; el.style.transform = 'translateY(18px)';
    el.style.transition = 'opacity .6s ease, transform .6s ease'; obs.observe(el);
  });
})();
