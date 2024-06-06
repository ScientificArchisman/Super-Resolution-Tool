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

document.addEventListener("DOMContentLoaded", function () {
    const darkModeToggle = document.getElementById("dark-mode-toggle");
    const body = document.body;
  
    darkModeToggle.addEventListener("change", function () {
      body.classList.toggle("night-mode");
    });
  });
  
  function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
      const output = document.getElementById("uploaded-image");
      output.src = reader.result;
      output.style.display = "block";
    };
    reader.readAsDataURL(event.target.files[0]);
  }
  
  function submitForm() {
    const form = document.getElementById("upload-form");
    const formData = new FormData(form);
    const resultImage = document.getElementById("result-image");
  
    fetch("/upload", {
      method: "POST",
      body: formData
    })
      .then(response => response.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        resultImage.src = url;
        resultImage.style.display = "block";
      })
      .catch(error => console.error("Error:", error));
  }

