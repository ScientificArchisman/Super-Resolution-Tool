function previewImage(event) {
    const input = event.target;
    const reader = new FileReader();
    
    reader.onload = function() {
        const uploadedImage = document.getElementById('uploaded-image');
        uploadedImage.src = reader.result;
        uploadedImage.style.display = 'block';
    };
    
    if (input.files && input.files[0]) {
        reader.readAsDataURL(input.files[0]);
    }
}

function submitForm() {
    const model = document.querySelector('input[name="model"]:checked').value;
    const input = document.getElementById('upload-image');

    if (input.files.length > 0) {
        const formData = new FormData();
        formData.append('image', input.files[0]);
        formData.append('model', model);

        fetch('/predict', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
          .then(data => {
              document.getElementById('result').innerText = `Prediction: ${data.prediction}`;
          })
          .catch(error => console.error('Error:', error));
    } else {
        alert('Please upload an image.');
    }
}

document.addEventListener('DOMContentLoaded', function () {
    const toggleSwitch = document.getElementById('switch-1');
    toggleSwitch.addEventListener('change', function () {
        document.getElementById("background").classList.toggle("on");
        document.body.classList.toggle('night-mode');
    });
});
