const obs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      obs.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.fade-in').forEach(el => obs.observe(el));

document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// REFERENCES TOGGLE
const refBtn = document.querySelector('.view-more-refs');
const moreRefs = document.querySelector('.more-refs');

if (refBtn && moreRefs) {
  refBtn.addEventListener('click', () => {
    const isHidden = moreRefs.classList.contains('hidden');
    
    if (isHidden) {
      // Show
      moreRefs.classList.remove('hidden');
      setTimeout(() => {
        moreRefs.classList.add('visible');
      }, 10);
      refBtn.textContent = 'Show Less ↑';
    } else {
      // Hide
      moreRefs.classList.remove('visible');
      setTimeout(() => {
        moreRefs.classList.add('hidden');
      }, 300);
      refBtn.textContent = 'View All References →';
    }
  });
}
