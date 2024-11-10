let uploadedFile;

function previewImage(event) {
  const file = event.target.files[0];
  if (file) {
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = function (e) {
      const uploadedImage = document.getElementById('uploadedImage');
      uploadedImage.src = e.target.result;
      uploadedImage.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }
}

async function classifyImage() {
  if (!uploadedFile) {
    alert("Please upload an image first!");
    return;
  }

  const formData = new FormData();
  formData.append('image', uploadedFile);

  const response = await fetch('/classify', {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();
  document.getElementById("result").textContent = `Outfit Type: ${result.class}`;
  document.getElementById("accuracy").textContent = `Accuracy: ${result.accuracy}`;
  document.getElementById("loss").textContent = `Loss: ${result.loss}`;
}
