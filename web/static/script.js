document.addEventListener('DOMContentLoaded', () => {
    let generateBtn = document.getElementById('generate-btn');
    let stopBtn     = document.getElementById('stop-btn');
    const videoStream = document.getElementById('video-stream');
    let questionField = document.getElementById('question');

    generateBtn.addEventListener('click', () => {
        fetch('/question')
        .then(response => response.json())
        .then(data => {
            questionField.textContent = data.question;
            videoStream.src = "/start";
        });
    });
    stopBtn.addEventListener('click', () => {
        questionField.textContent = 'Answer the question by nodding your head..Are you ready?';
        videoStream.src = src="/static/curious_default.jpg";

    });
});
