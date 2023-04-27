
function escapeHTML(unsafeText)
{
    let div = document.createElement('div');
    div.innerText = unsafeText;
    return div.innerHTML;
}

// Endpoint for websocket:
var endpoint = "ws://localhost:9000/chat";
var ws = new WebSocket(endpoint);

// Default subtitle:
default_subtitle = " Ask a question, ask for a widget, ask to process images, or control the napari viewer ! ";

// Receive message from server and process it:
ws.onmessage = function (event)
{

    // Message list element:
    var messages = document.getElementById('messages');

    // JSON parsingof message data:
    var data = JSON.parse(event.data);

    // Log event on the console for debugging:
    console.log("__________________________________________________________________");
    console.log("data.sender ="+data.sender+'\n');
    console.log("data.type   ="+data.type+'\n');
    console.log("data.message="+data.message+'\n');

    // Message received from agent:
    if (data.sender === "agent")
    {
        // start message:
        if (data.type === "start")
        {
            // Set subtitle:
            var header = document.getElementById('header');
            header.innerHTML = "Thinking...";

        }
        // agent is typing:
        else if (data.type === "typing")
        {
            // Set subtitle:
            var header = document.getElementById('header');
            header.innerHTML = 'Typing...';
        }
        // action message:
        else if (data.type === "action")
        {

             // Set subtitle:
            var header = document.getElementById('header');
            header.innerHTML = "Using a tool...";

            // Create a new message entry:
            var div = document.createElement('div');
            div.className = 'server-message';
            var p = document.createElement('p');

            // Add this new message into message list:
            div.appendChild(p);
            messages.appendChild(div);

            // Current (last) message:
            var p = messages.lastChild.lastChild;

            // Set background color:
            p.parentElement.className = 'action-message';

            // Parse markdown and render as HTML:
            p.innerHTML = "<strong>" + "Omega: " + "</strong>" + marked.parse(data.message)

        }
        // Tool end:
        else if (data.type === "tool_end")
        {
            // Current (last) message:
            var p = messages.lastChild.lastChild;

            // Set background color:
            //p.parentElement.className = 'server-message';

            // markdown:
            message = data.message

            // Parse markdown and render as HTML:
            p.innerHTML += marked.parse(data.message)

        }
        // action message:
        else if (data.type === "tool_result")
        {

            // Current (last) message:
            var p = messages.lastChild.lastChild;

            // Parse markdown and render as HTML:
            p.innerHTML += "<br>"+marked.parse(data.message);
        }
        // end message, this is sent once the agent has a final response:
        else if (data.type === "final")
        {
            // Create a new message entry:
            var div = document.createElement('div');
            div.className = 'server-message';
            var p = document.createElement('p');

            // Add message to message list:
            div.appendChild(p);
            messages.appendChild(div);

            // Current (last) message:
            var p = messages.lastChild.lastChild;

            // Set background color:
            p.parentElement.className = 'server-message';

            // Parse markdown and render as HTML:
            p.innerHTML = "<strong>" + "Omega: " + "</strong>" + marked.parse(data.message)

            // Reset subtitle:
            var header = document.getElementById('header');
            header.innerHTML = default_subtitle;

            // Reset button text and state:
            var button = document.getElementById('send');
            button.innerHTML = "Send";
            button.disabled = false;
        }
        // Error:
        else if (data.type === "error")
        {
            // Reset subtitle:
            var header = document.getElementById('header');
            header.innerHTML = default_subtitle;

            // Reset button text and state:
            var button = document.getElementById('send');
            button.innerHTML = "Send";
            button.disabled = false;

            // Create a new message entry:
            var div = document.createElement('div');
            div.className = 'server-message';
            var p = document.createElement('p');

            // Add message to message list:
            div.appendChild(p);
            messages.appendChild(div);

            // Display error message:
            var p = messages.lastChild.lastChild;
            p.innerHTML = "<strong>" + "Omega: " + "</strong>" + marked.parse(data.message)

            // Set background color:
            p.parentElement.className = 'error-message';
        }
    }
    // Message received from user:
    else if (data.sender === "user")
    {
        // Create new message entry:
        var div = document.createElement('div');
        div.className = 'client-message';
        var p = document.createElement('p');

        // Set default (empty) message:
        p.innerHTML = "<strong>" + "You: " + "</strong>";
        p.innerHTML += marked.parse(data.message);

        // Add message to message list:
        div.appendChild(p);
        messages.appendChild(div);
    }
    // Scroll to the bottom of the chat
    messages.scrollTop = messages.scrollHeight;
};


// Send message to server
function sendMessage(event)
{
    event.preventDefault();
    var message = document.getElementById('messageText').value;
    if (message === "") {
        return;
    }
    ws.send(message);
    document.getElementById('messageText').value = "";

    // Turn the button into a loading button
    var button = document.getElementById('send');
    button.innerHTML = "Loading...";
    button.disabled = true;
}