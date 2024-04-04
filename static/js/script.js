//點選選擇頭髮後觸發
const hair_menu = document.querySelector(".hair-menu");
const contaner1_2 = document.querySelector(".hair-list-view");
// const contaner1 = document.querySelector(".creat-input");
hair_menu.addEventListener("click", () => {
    contaner1_2.classList.toggle("active");
    // contaner1.classList.toggle("active");
  })

//選擇圖片後觸發
window.onload = function(){
    var obj_lis = document.getElementsByClassName("hair-view")
    var image = document.getElementsByClassName("select_img")[0];
    for(i=0;i<obj_lis.length;i++){
        obj_lis[i].onclick = function(){
            image.src = this.getAttribute("src");
            contaner1_2.classList.toggle("active");
            // contaner1.classList.toggle("active");
        }
    }
}

var ss = 0

//確定並送出頭髮圖片
function send_img_to_python()
{
    ss += 1;
    console.log(ss);
    if ($('.select_img').attr('src') != 'static/image/default.jpg'){

        $.ajax({
            url: "/return_hair",
            type: 'POST',
            data:{"select_img":$('.select_img').attr('src'),
                "my_img":$('.output1').attr('src')},
            dataType: "json",
            beforeSend: function () {
                // 顯示載入指示符或任何其他視覺提示，表示正在處理請求
                $('.loading-finish').hide();

                $('.loading-blank').hide();

                $('.loading-spinner').show();

                $('.loading-spinner1').show();

                setTimeout(function() {
                  $('.loading-spinner2').show();
                }, 500); // 1秒延遲顯示 loading-spinner2

                setTimeout(function() {
                  $('.loading-spinner3').show();
                }, 1000); // 2秒延遲顯示 loading-spinner3
              },
            success: function (response) {
                console.log("success");
                //更改背景圖片(換成沒頭髮)
                $('.img_background').attr('src',response['no_hair_img']);
                //取得圖片後渲染上去
                $('.img_hair').attr('src',response['hair_img']);
                //設定長寬、預設位置
                $('.img_hair').css({'width' : response['hair_img_x_size'] * 2 + 'px',
                                     'height' : response['hair_img_y_size'] * 2 + 'px',
                                     'left' : response['x']*2 + 'px',
                                     'top' : response['y']*2 + 'px',
                                     'transform':"rotate(0deg)"});

                $('.slider_x').attr({"max":response['max_x'],"value":response['x']});
                $(".slider_x_value").text(response['x']);
                $('.slider_y').attr({"max":response['max_y'],"value":response['y']});
                $(".slider_y_value").text(response['y']);
                $('.slider_size_x').attr({"min":1,"max":256 , "value":response['hair_img_x_size']});
                $('.slider_size_x_value').text(response['hair_img_x_size']);
                $('.slider_size_y').attr({"min":1,"max":256 , "value":response['hair_img_y_size']});
                $('.slider_size_y_value').text(response['hair_img_y_size']);
                $('.slider_rotate').attr({"min":-10,"max":10,'value':"0"});
                //放在上面不知道為甚麼沒工作
                document.getElementsByClassName("slider_rotate")[0].value = "0";
                $('.slider_rotate_value').text($('.slider_rotate').attr('value'));
                $('.certain').toggle(true);

                // 請求完成後隱藏載入指示符
                $('.loading-spinner').hide();
                $('.loading-spinner1').hide();
                $('.loading-spinner2').hide();
                $('.loading-spinner3').hide();
                $('.loading-finish').show();
            },
            error: function (xhr, status, error) {
                console.log("錯誤");

                // 請求完成後隱藏載入指示符（錯誤情況下也要隱藏）
                $('.loading-spinner').hide();
                $('.loading-spinner1').hide();
                $('.loading-spinner2').hide();
                $('.loading-spinner3').hide();
                $('.loading-blank').show();

                // 如有需要，處理錯誤
              }
        });
    }
}

//控制頭髮左右
$('.slider_x').change(function(){
    $(".img_hair").css('left' , this.value*2+"px")
    $(".slider_x_value").text(this.value);
    $('.slider_x').attr('value',this.value);
});

//控制頭髮上下
$('.slider_y').change(function(){
    $(".img_hair").css('top' , this.value*2+"px")
    $(".slider_y_value").text(this.value);
    $('.slider_y').attr('value',this.value);
});

//控制頭髮size_x
$('.slider_size_x').change(function(){
    $(".img_hair").css('width' , this.value*2+"px")
    $('.slider_size_x_value').text(this.value);
    $('.slider_size_x').attr('value',this.value);
});

//控制頭髮size_y
$('.slider_size_y').change(function(){
    $(".img_hair").css('height' , this.value*2+"px")
    $('.slider_size_y_value').text(this.value);
    $('.slider_size_y').attr('value',this.value);
});

//控制頭髮旋轉
$('.slider_rotate').change(function(){
    $('.img_hair').css('transform',"rotate("+this.value+"deg)");
    $('.slider_rotate_value').text(this.value);
    $('.slider_rotate').attr('value',this.value);
});

//送出調整過後的資料
$('.certain').click(function(){
    $.ajax({
        url: "/receive_all_data",
        type: 'POST',
        data:{"pos_x" : $('.slider_x').attr('value'),
              "pos_y" : $('.slider_y').attr('value'),
              "img_width" : $('.slider_size_x').attr('value'),
              "img_height" : $('.slider_size_y').attr('value'),
              "rotate_value" : $('.slider_rotate').attr('value')},
        dataType: "json",
        beforeSend: function () {
            // 顯示載入指示符或任何其他視覺提示，表示正在處理請求
            setTimeout(function() {
                $('.loading-blank-x').hide();
                $('.loading-blank-0').show();
            }, 400);
            setTimeout(function() {$('.loading-blank-1').show();}, 600);
            setTimeout(function() {$('.loading-blank-2').show();}, 800);
            setTimeout(function() {$('.loading-blank-3').show();}, 1000);
            setTimeout(function() {$('.loading-blank-4').show();}, 1200);
            setTimeout(function() {$('.loading-blank-5').show();}, 1400);
            setTimeout(function() {$('.loading-blank-6').show();}, 1600);
            setTimeout(function() {$('.loading-blank-7').show();}, 1800);
            setTimeout(function() {$('.loading-blank-8').show();}, 2000);
            setTimeout(function() {$('.loading-blank-9').show();}, 2200);
          },
        success: function (response){
            console.log("ok");
            //切換到顯示圖片
            $('.creat').toggle(false);
            const show_creat = document.querySelector(".show_creat");
            show_creat.classList.toggle("active");
            //設定圖片
            ori_img = response['ori_img'];
            sketch_img = response['sketch_img'];

            $('.canvas').attr('src' , 'data:image/png;base64,'+ori_img);
        },
        error: function(response){
            console.log(response);
        }
    })
});

$('.test').click(function(){
    $('.creat').toggle(false);
    const show_creat = document.querySelector(".show_creat");
    show_creat.classList.toggle("active");
})

//儲存處理好的圖片
var ori_img = 0;
var sketch_img = 0;
var download_base64 = 0;

$(document).ready(function(){
    $('.opt1').click(function(){
        if(ori_img != 0){
            $('.canvas').attr('src' , 'data:image/png;base64,'+ori_img);
            download_base64 = ori_img;
        }
    });

    $('.opt2').click(function(){
        if(sketch_img != 0){
            $('.canvas').attr('src' , 'data:image/png;base64,'+sketch_img);
            download_base64 = sketch_img;
        }
    });

    $('.pre_page').click(function(){
        history.go(0);
    });

 });

 function downloadBase64img(){
    if(download_base64 != 0){
        const linkSource = 'data:image/png;base64,'+download_base64;
        const downloadLink = document.createElement("a");
        downloadLink.href = linkSource;
        downloadLink.download = "test.jpg";
        downloadLink.click();
    }
}