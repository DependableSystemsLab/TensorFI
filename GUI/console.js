const fs = require('fs');
const yaml = require('js-yaml');
const readline = require("readline");
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

var errorRate;
// Prompts user for the file path of the tensorflow model to run
rl.question("TensorFlow File Path: ", function(filePath) {
    rl.question("Log Directory: ", function(logDir) {
        // Promps the user for an injection mode
        rl.question("Injection Mode: ", function(injectMode) {
            // Prompts the user for Fault Types
            rl.question("Scalar Fault Type: ", function(scalarFaultType) {
                rl.question("Tensor Fault Type: ", function(tensorFaultType) {
                    // Writes user input into a config file
                    let data = {
                        ScalarFaultType: scalarFaultType,
                        TensorFaultType: tensorFaultType,
                        InjectMode: injectMode,
			Instances: null,
			Ops: ['ALL = 1.0']
                    };
                    let yamlStr = yaml.safeDump(data);
                    fs.writeFileSync('./config.yaml', yamlStr, 'utf8');

                    // Runs a python process with the required variables
                    var spawn = require("child_process").spawn; 

                    var parcer = spawn('python',["./parser.py", 
                                       './config.yaml', 
                                       logDir, 
                                       filePath]);

                    parcer.stdout.on('data', function(data) {
                        var test = spawn('python',["runFile.py"]);

                        test.stdout.on('data', function(data) { 
                            errorRate = data.toString();
                            rl.close();
                        });
                    });
                });
            });
        });
    });
});

rl.on("close", function() {
    console.log(errorRate);
    let data = yaml.safeLoad(fs.readFileSync('./config.yaml', 'utf8'));

    if (data["Instances"] != null) {
        data["Ops"] = [];
        for (var instance in data["Instances"]) {
            op = instance.split(" = ")[0];
            rl.question(op + " (0.0 - 1.0): ", function(percent) {
                data["Ops"].push(op + " = " + percent);
            });
        }
    }
    let yamlStr = yaml.safeDump(data);
    fs.writeFileSync('./config.yaml', yamlStr, 'utf8');

    process.exit(0);
});

