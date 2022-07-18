
//////////////////////////////////////////////////////////////////////////////////////////
// Definitions
//////////////////////////////////////////////////////////////////////////////////////////
    #include <IRremote.h> // Package for the remote

        
    #define SENSORPIN 7 // Input/Sensor of the IR Beam
    #define REMOTEPIN 10 // Input of the remote receiver

    #define TTL_AMP 9
    #define TTL_OPTO 11

    #define LED_MOCAP 8 //Output LED to sync MOCAP
    
    IRrecv irrecv(REMOTEPIN); //Pin for the package of remote sensor
    
    
    // variables will change:
    int sensorState = 0, lastState=0;         // variable for reading the sensor status
    decode_results results; // Variable for the result of the remote  

    int Record_Status = 0;
    
    //Add variables for opto frequency
    
//////////////////////////////////////////////////////////////////////////////////////////
// Setup
//////////////////////////////////////////////////////////////////////////////////////////
     
    void setup() {
      // initialize the LED pin as an output:
      pinMode(LED_MOCAP, OUTPUT);
      digitalWrite(LED_MOCAP, LOW);  

      // initialize the TTL pin as an output:

      pinMode(TTL_AMP, OUTPUT);
      digitalWrite(TTL_AMP, LOW);
      
      pinMode(TTL_OPTO, OUTPUT);
      digitalWrite(TTL_OPTO, LOW);

      // initialize the sensor pin as an input:
      pinMode(SENSORPIN, INPUT);         
      digitalWrite(SENSORPIN, HIGH);
      
      Serial.begin(9600);
      irrecv.enableIRIn();
//      irrecv.blink13(true);
      
    }

//////////////////////////////////////////////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////////////////////////////////////////////

void Triggered(){
  // The beam is triggered : Turn on opto
  for (int i = 0; i <= 500; i++) {
    digitalWrite(TTL_OPTO, HIGH);
    delay(5);
    digitalWrite(TTL_OPTO, LOW);
    delay(7.5);
    Serial.println(i);
  }
}
  


//////////////////////////////////////////////////////////////////////////////////////////
//Loop
//////////////////////////////////////////////////////////////////////////////////////////   
    void loop(){
      // read the state of the IR remote receiver value:
      sensorState = digitalRead(SENSORPIN);

     
      // check if the sensor beam is broken
      // if it is, the sensorState is LOW:
      if (sensorState == LOW) {
        Serial.println("Broken");
        Triggered();
        delay(2000);
      } 

      //Check if the Remote sensor detect Remote Button being pushed
      // If it is, send the button value to serial
      else if (irrecv.decode(&results)){
        irrecv.resume();
        Serial.println(results.value);
        if (results.value == 1286666973 && Record_Status ==0){
          Serial.println(results.value, HEX);
          Serial.println("Start");
          delay(500);
          digitalWrite(LED_MOCAP, HIGH);
          digitalWrite(TTL_AMP, HIGH);
          delay(5);
          digitalWrite(TTL_AMP, LOW);
          delay(500);
          digitalWrite(LED_MOCAP, LOW);
          Record_Status = 1;
        }
        
        if (results.value == 4294967295 && Record_Status ==1){
          Serial.println(results.value, HEX);
          Serial.println("Stop");
          Record_Status = 0;
        }
        
      }

      else if (irrecv.decode(&results)){
        irrecv.resume();

        
      }
      

      //Else, if the beam is not broken, say it, and wait
      if (sensorState && !lastState) {
        Serial.println("Unbroken");
      } 
//      if (!sensorState && lastState) {
//        Serial.println("Broken");
//      }
      lastState = sensorState;

    }
