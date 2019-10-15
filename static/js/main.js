$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
	//$('#pred_img').hide();
	$('#pred_img').empty();
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
		$('#btn-show').hide();
		$('#pred_img').hide();
        $('#result').text('');
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
				$('#btn-show').show();
                console.log('Success!');
            },
        });
		
    });
	
	
	// Display Image
    $('#btn-show').click(function () {
		//console.log(111111111111111111111111);
		$('#pred_img').empty();
		$(this).hide();
		$('#pred_img').show();
		
		//var img = document.createElement("img");
		//img.src = "static/preds/predicted_img.jpg?" + new Date().getTime();
		//
		//$('#pred_img').html(img); 
		
		
		//function ReplacingImage() {
		//	document.getElementById('pred_img').src='static/preds/predicted_img.jpg'
		//}
		var img = new Image();
		var div = document.getElementById('pred_img');
		img.src = "static/preds/predicted_img.jpg?" + new Date().getTime();
		img.onload = function() {
			div.innerHTML = '<img width="400px" height="320px" src="'+img.src+'" align="right" />'; 
			};
		   
		////img.parentNode.removeChild(img);
		
    });
	
});
