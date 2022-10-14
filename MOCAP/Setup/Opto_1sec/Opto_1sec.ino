
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
    digitalWrite(TTL_OPTO, HIGH);
    delay(5);
    digitalWrite(TTL_OPTO, LOW);
    delay(1000);
}
