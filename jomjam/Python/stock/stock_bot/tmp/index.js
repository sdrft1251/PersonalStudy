// Import express and request modules
var express = require('express');
var request = require('request');

// Store our app's ID and Secret. These we got from Step 1. 
// For this tutorial, we'll keep your API credentials right here. But for an actual app, you'll want to  store them securely in environment variables. 
var clientId = '1617107260080.1578674369335';
var clientSecret = 'bf89851835b2cf903ac72662c6f22c17';

// Instantiates Express and assigns our app variable to it
var app = express();


// Again, we define a port we want to listen to
const PORT = 4390;


// Lets start our server
app.listen(PORT, function () {
    //Callback triggered when server is successfully listening. Hurray!
    console.log("Example app listening on port " + PORT);
});


// This route handles GET requests to our root ngrok address and responds with the same "Ngrok is working message" we used before
app.get('/', function (req, res) {
    res.send('Ngrok is working! Path Hit: ' + req.url);
});

// This route handles get request to a /oauth endpoint. We'll use this endpoint for handling the logic of the Slack oAuth process behind our app.
app.get('/oauth', function (req, res) {
    // When a user authorizes an app, a code query parameter is passed on the oAuth endpoint. If that code is not there, we respond with an error message
    if (!req.query.code) {
        res.status(500);
        res.send({ "Error": "Looks like we're not getting code." });
        console.log("Looks like we're not getting code.");
    } else {
        // If it's there...

        // We'll do a GET call to Slack's `oauth.access` endpoint, passing our app's client ID, client secret, and the code we just got as query parameters.
        request({
            url: 'https://slack.com/api/oauth.access', //URL to hit
            qs: { code: req.query.code, client_id: clientId, client_secret: clientSecret }, //Query string data
            method: 'GET', //Specify the method

        }, function (error, response, body) {
            if (error) {
                console.log(error);
            } else {
                res.json(body);

            }
        })
    }
});

// Route the endpoint that our slash command will point to and send back a simple response to indicate that ngrok is working
app.post('/stock_info', function (req, res) {

    const { PythonShell } = require("python-shell");
    let options = {
        scriptPath: "C:\\Users\\sdrft\\source\\repos\\My_stock_project",
    };
    PythonShell.run("connecting_test.py", options, function(err, data) {
        if (err) throw err;
    });
    res.send("Wait for seconds...");
});

app.post("/event", function (req, res, next) {
    let payload = req.body;
    res.sendStatus(200);

    if (payload.event.type === "app_mention") {
        if (payload.event.text.includes("tell me a joke")) {
            // Make call to chat.postMessage using bot's token
        }
    }
    if (payload.event.type === "message") {
        let response_text;
        if (payload.event.text.includes("Who's there?")) {
            response_text = "A bot user";
        }
        if (payload.event.text.includes("Bot user who?")) {
            response_text = "No, I'm a bot user. I don't understand jokes.";
        }
        if (response_text !== undefined) {
            // Make call to chat.postMessage sending response_text using bot's token
        }
    }
});
