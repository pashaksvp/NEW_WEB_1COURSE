document.addEventListener('DOMContentLoaded', () => {

    gsap.registerPlugin(ScrollTrigger, ScrollSmoother);

    ScrollSmoother.create({
        wrapper: '.wrapper',
        content: '.content',
        smooth: 5,
        effects: true
    });  

    gsap.fromTo('.accordion__img', {x: -250,opacity: 0}, {
        opacity: 1, x: 50,
        scrollTrigger: {
            trigger: '.accordion__img',
            end: '0',
            scrub: true
        }
    })
    
    gsap.fromTo('.accordion__list', {x: 250,opacity: 0}, {
        opacity: 1, x: -100,
        scrollTrigger: {
            trigger: '.accordion__list',
            end: '0',
            scrub: true
        }
    })
})
$(function () {
        
    'use strict';

    function accordion() {
        $('.accordion .accordion__item').on('click', function () {
            const timeAnim = 400;
            $('.accordion .accordion__item').removeClass("active").css({ 'pointer-events' : 'auto'});
            $(this).addClass("active").css({ 'pointer-events' : 'none'});
            
            $('.accord__img').removeClass("active");
            let id = $(this).data('id');
            $('#' + id + '-img').addClass("active");
        });
        
    }
    accordion();
});