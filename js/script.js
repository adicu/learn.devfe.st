function showEmail(){$("#success").hide(),$("#mce-EMAIL").fadeIn(),$("#signup").hide(),$("#submit").show()}function submitEmail(){var o=/^[A-Z0-9._%+-]+@([A-Z0-9-]+\.)+[A-Z]{2,4}$/i,i=$("#mce-EMAIL").val();o.test(i)?($("form[name=mc-embedded-subscribe-form]").submit(),$("#submit").hide(),$("#mce-EMAIL").hide(),$("#success").show()):alert("Incorrect Email. Please Try Again.")}function showVideo(){$("#viewvideo").hide(),$("#gif2015").hide(),$("#vid2015").fadeIn(),$(".transparent").click(function(){$("#viewvideo").show(),$("#gif2015").show(),$("#vid2015").fadeOut()})}function applyNavScroll(){var o=$("#schedule").offset().top;console.log(o),$(window).scroll(function(){$(this).scrollTop()>o?$(".nav").css({position:"fixed"}):$(".nav").css({position:"absolute"})})}function showNav(){$("#exit").show(),$(".unhide-nav").show(),$("#menu").hide()}function hideNav(){$(".unhide-nav").hide(),$("#menu").show(),$("#exit").hide()}function showPopup(){$(".popup-content").show(),$("#welcome-intro").hide()}function hidePopup(){$(".popup-content").hide();var o=$("#playerid").attr("src");$("#playerid").attr("src",""),$("#playerid").attr("src",o),$("#welcome-intro").show()}$(document).ready(function(){applyNavScroll();$(window).width();$(window).resize(applyNavScroll);var o="";$(".scroll").click(function(i){this.hash;if(o!=this.hash){i.preventDefault();var e=0;e=$(this.hash).offset().top>$(document).height()-$(window).height()?$(document).height()-$(window).height():$(this.hash).offset().top;var t=1e3;$("html,body").animate({scrollTop:e},t,"swing"),o=this.hash}}),$(window).scroll(function(){var o=$(document.body),i=o.height();$("#globe").css({transform:"translateX(+50%) translateY(+50%) rotate("+o.scrollTop()/i*360+"deg)"}),100*(o.scrollTop()/i-1)+25>0&&$("#airplane").css({right:100*(o.scrollTop()/i-1)+25+"%"}),$(".hideme").each(function(o){var i=$(this).offset().top+$(this).outerHeight(),e=$(window).scrollTop()+$(window).height();e>i&&$(this).animate({opacity:"1"},500)})}),$("#moreinfo").click(function(){$("html, body").animate({scrollTop:$("#info").offset().top},1e3)}),$("#main").click(function(){$("html, body").animate({scrollTop:$("#landing").offset().top},1e3)})});