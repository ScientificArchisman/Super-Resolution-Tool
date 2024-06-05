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
    const width = document.getElementById('width').value;
    const height = document.getElementById('height').value;

    if (input.files.length > 0) {
        const formData = new FormData();
        formData.append('image', input.files[0]);
        formData.append('model', model);
        formData.append('width', width);
        formData.append('height', height);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const resultImage = document.getElementById('result-image');
            resultImage.src = `data:image/png;base64,${data.image}`;
            resultImage.style.display = 'block';
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
