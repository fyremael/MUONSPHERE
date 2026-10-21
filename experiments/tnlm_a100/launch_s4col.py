import pathlib,subprocess,sys
log=pathlib.Path('/content/gcl_fsm_s4_col')
log.mkdir(parents=True,exist_ok=True)
f=(log/'controller.log').open('w',buffering=1)
p=subprocess.Popen([sys.executable,'/content/gcl_tnlm_s4col_controller.py'],
    stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('FSM_REPLAY_CONTROLLER_SPAWNED',p.pid,flush=True)
