
//////////////////////////////////////////////////////////////////////////////////////////
// Definitions
//////////////////////////////////////////////////////////////////////////////////////////
   
    #define SENSORPIN 7 // Input/Sensor of the IR Beam
    #define REMOTEPIN 10 // Input of the remote receiver

    #define TTL_AMP 9
    #define TTL_OPTO 11

    #define LED_MOCAP 8 //Output LED to sync MOCAP
    

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
      
      //Serial.begin(9600);
    }

//////////////////////////////////////////////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////////////////////////////////////////////

void brust(int stim_duration,float freq,int pulse_duration){
          for (int i = 1; i <= stim_duration*freq; i++) {
          digitalWrite(TTL_AMP, HIGH);
          digitalWrite(TTL_OPTO, HIGH);
          delay(pulse_duration);
          digitalWrite(TTL_AMP, LOW);
          digitalWrite(TTL_OPTO, LOW);
          float off_time = 1/freq*1000;
          //Serial.println(off_time-pulse_duration);
          delay(off_time-pulse_duration);
          }
}


//////////////////////////////////////////////////////////////////////////////////////////
//Loop
//////////////////////////////////////////////////////////////////////////////////////////   
    void loop(){
      float stim_duration = 10;//in seconds
      float freq = 1;//in Hz
      float pulse_duration = 100;//in ms  
      float time_between_trans = 1;//in s
      
      brust(stim_duration,freq,pulse_duration);
      delay(time_between_trans);
          
    }
