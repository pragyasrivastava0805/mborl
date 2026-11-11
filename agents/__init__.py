# Baselines (Model-free)
from agents.iql import IQLAgent
from agents.cql import CQLAgent
from agents.rebrac import ReBRACAgent
from agents.nstep_iql import NStepIQLAgent
from agents.gciql import GCIQLAgent
from agents.qrl import QRLAgent
from agents.crl import CRLAgent
from agents.ngcsacbc import NGCSACBCAgent
from agents.hiql import HIQLAgent

# Baselines (Model-based)
from agents.mbpo import MBPOAgent
from agents.mopo import MOPOAgent
from agents.morel import MOReLAgent
from agents.reverse_morel import ReverseMOReLAgent
from agents.reverse_morel_anchor_bootstrap import ReverseMOReLAnchorBootstrapAgent
from agents.reverse_rollout_high_freq import ReverseRolloutHighFreqAgent
from agents.forward_morel_iql import ForwardMOReLIQLAgent
from agents.forward_morel_cql import ForwardMOReLCQLAgent
from agents.reverse_morel_cql import ReverseMOReLCQLAgent
from agents.forward_morel_rebrac import ForwardMOReLReBRACAgent
from agents.reverse_morel_rebrac import ReverseMOReLReBRACAgent
from agents.combo import COMBOAgent
from agents.fmpc import FMPCAgent
from agents.leq import LEQAgent
from agents.mobile import MOBILEAgent

# Other agents
from agents.mac import MACAgent
from agents.mbrs_ac import ACMBRSAgent
from agents.mbfql import MBFQLAgent
from agents.model_ac import ACModelAgent

agents = dict(
    iql=IQLAgent,
    cql=CQLAgent,
    rebrac=ReBRACAgent,
    nstep_iql=NStepIQLAgent,
    gciql=GCIQLAgent,
    qrl=QRLAgent,
    crl=CRLAgent,
    ngcsacbc=NGCSACBCAgent,
    hiql=HIQLAgent,

    mbpo=MBPOAgent,
    mopo=MOPOAgent,
    morel=MOReLAgent,
    reverse_morel=ReverseMOReLAgent,
    reverse_morel_anchor_bootstrap=ReverseMOReLAnchorBootstrapAgent,
    reverse_rollout_high_freq=ReverseRolloutHighFreqAgent,
    forward_morel_iql=ForwardMOReLIQLAgent,
    forward_morel_cql=ForwardMOReLCQLAgent,
    reverse_morel_cql=ReverseMOReLCQLAgent,
    forward_morel_rebrac=ForwardMOReLReBRACAgent,
    reverse_morel_rebrac=ReverseMOReLReBRACAgent,
    combo=COMBOAgent,
    fmpc=FMPCAgent,
    leq=LEQAgent,
    mobile=MOBILEAgent,

    mac=MACAgent,
    mbrs_ac=ACMBRSAgent,
    mbfql=MBFQLAgent,
    model_ac=ACModelAgent,
)
