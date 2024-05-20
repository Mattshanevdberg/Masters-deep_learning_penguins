import subprocess

def run_script(script_name):
    # Call the second script
    result = subprocess.run(['python', script_name], capture_output=True, text=True)

    # Print output and error if any
    print("Output:", result.stdout)
    print("Errors:", result.stderr)

    # Check the return code
    if result.returncode == 0:
        print("Script executed successfully")
    else:
        print("Script failed with return code", result.returncode)

# Call the function

#run_script('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_scn.py')

#run_script('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_ptn.py')

#run_script('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_scs.py')

#run_script('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_pts.py')

run_script('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_scm.py')

run_script('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_ptm.py')
