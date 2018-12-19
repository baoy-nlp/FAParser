import os
import subprocess
evalb_dir = "../EVALB"
evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
evalb_program_path = os.path.join(evalb_dir, "./evalb")
temp_ref_path = os.path.join(evalb_dir, "sample/sample.gld")
temp_pred_path = os.path.join(evalb_dir, "sample/sample.tst")
temp_eval_path = os.path.join(evalb_dir, "sample/sample.score")

command = r"{} -p {} {} {} > {}".format(
    evalb_program_path,
    evalb_param_path,
    temp_ref_path,
    temp_pred_path,
    temp_eval_path)
subprocess.run(command,shell=True)
# os.system(command)
