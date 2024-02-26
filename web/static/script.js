document.addEventListener('DOMContentLoaded', () => {
    let generateBtn   = document.getElementById('generate-btn');
    let submitBtn     = document.getElementById('submit-btn');
    const videoStream = document.getElementById('video-stream');
    let questionField = document.getElementById('question');
    let socket        = io.connect('http://' + document.domain + ':' + location.port);
    let loggedQuestions = []
    let logTableBody  = document.getElementById('log-table-body');

    let clickCount_play = 0;
    let clickCount_log = 0;

    generateBtn.addEventListener('click', () => {
        fetch('/question')
        .then(response => response.json())
        .then(data => {
            questionField.textContent = data.question;
            videoStream.src = "/start";
        });
        clickCount_play++;
        if (clickCount_play === 1) {
            generateBtn.textContent = "Play Again";
        }
        submitBtn.classList.remove('green');

    });

    submitBtn.addEventListener('click', () => {
        questionField.textContent = 'Answer the question by nodding your head..Are you ready?';
        videoStream.src           = src="/static/curious_default.jpg";
        submitBtn.classList.add('green');
        clickCount_log++;
        if (clickCount_log === 1) {
            submitBtn.textContent = "Reset";
        }
        socket.on('log', function(data) {        
            // Split the log message into question, gesture, and timestamp
            let parts     = data.split(' - ');
            let question  = parts[0].split(': ')[1];
            let gesture   = parts[1].split(': ')[1];
            let timestamp = parts[2].split(': ')[1];
            
            if (!loggedQuestions.includes(question)){
    
                let newRow        = logTableBody.insertRow();
                let questionCell  = newRow.insertCell(0);
                let gestureCell   = newRow.insertCell(1);
                let timestampCell = newRow.insertCell(2);
            
                // Set the table cell values
                questionCell.textContent  = question;
                gestureCell.textContent   = gesture;
                timestampCell.textContent = timestamp;
    
                loggedQuestions.push(question);
    
            }
        });

    });

});
