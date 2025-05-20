## bethLISA

Note: main_notebook contains a code tutorial/explanation on how to use the simulation along with running the q-transform code on TDI channels.

<ins>requirements.txt:</ins> all the modules needed for the code to run, just pip install after cloning the repository

<ins>Structure:</ins>

bethLISA
- lisa_glitch_simulation
  - final_tdi_outputs: where all the TDI channels created by the simulation live
  - input_output: where the in-between files containing injection information about the glitches live
  - simulate_glitches: the lisa glitch simulation
  - testing: all the config files and testing files
- q_transform: where the q-transform program is

### lisa_glitch_simulation

#### Instructions / Info

This is a section of my project on testing glitch detection algorithms. The program here simulates LISA glitches based on what was detected in LISAPathfinder. The user can decide inputs such as amplitude range, beta range (time dampening factor), time of injection distribution type (equally spaced or Poisson), and others such as time length etc. 
It was decided to test on a new set of data as the detection methods that were being compared were not showing a significant difference in detection ability when initially just using the LDC 2b-Spritz data-set. 

<ins>simulate_glitches program</ins>

Creates a time series of LISA data containing sine-gaussian glitches. Two main parts to the program: make_glitch, creates a .h5 glitch file based off user entered parameters along with a .txt file showing all the glitches in the data and inject_glitch, which takes the output glitch files from make_glitch and simulates the LISA spacecraft then calculates the TDI channels. 

#### Files in detail (the important ones)

<ins>make_glitch:</ins> 
Can be run alone directly from the command line (make sure to be in the /simulate_glitches/ directory)

Methods:
- main(args=None, testing=False): organises and runs all the functions below
- init_cl(testing=None): initialise the command line arguments with argparse
- use_testing_inputs(args, glitch_type): if we are using testing inputs (the defaults inputs of CL arguments) this sets up the parameters dictionary (params_dict) with those inputs
- simulate_glitches(glitch_type, glitch_amp_type, glitch_beta_type, params, testing=False): simulate the glitches using the lisaglitch module and save to a .h5 file (will be used as input for inject_glitch), also creates and saves a .txt file of all the created glitches and their parameters such as beta, level, and time of injection

<ins>inject_glitch:</ins> 
Can be run alone directly from the command line (make sure to be in the /simulate_glitches/ directory)

Methods:
- main(glitch_file_h5=None, glitch_file_txt=None, glitch_file_op=None): organises which files to pull glitches from then runs all the functions below 
- init_cl(): initialise the command line arguments with argparse
- init_inputs(): initialise the inputs variables from both .h5 file and .txt file
- simulate_lisa(glitch_file, glitch_inputs, noise=True): using the lisainstrument module simulate a lisa instrument with noise and using the input glitch file to inject glitches into the data
- tdi_channels(i, channels, inputs, tdi_names): takes the simulated instrument and constructs the TDI channels using PyTDI

Notes: to run just this file from the command line, uncomment the __name__ == "__main__": section at the end of the file

### Q-transform

<ins>Files:</ins>

The only file of interest is search.py, this file completes the whole q-transform search on TDI data, including clustering. I modified the code such that we were able to use it for glitch detection. The largest change would be that the TDI data is segmented into 24hr segments. Beforehand, when using the full TDI data lower amplitude glitches were missed as the q-transform was ‘focusing’ on the louder ones. By segmenting the data this issue is less present as each section won’t have such a large concentration of loud glitches. There were no other massive changes apart from not using refine_mismatch as it’s not needed for glitches. 
Note: When it saves it doesn’t save in the segmented version, the segments overlap by 12hr and the overlap is averaged to get the scan covering the whole TDI channel.
