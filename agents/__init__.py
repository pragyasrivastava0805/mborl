# Baselines (Model-free)
from agents.iql import IQLAgent
from agents.cql import CQLAgent
from agents.rebrac import ReBRACAgent
from agents.nstep_iql import NStepIQLAgent
from agents.gciql import GCIQLAgent
from agents.qrl import QRLAgent
from agents.crl import CRLAgent
from agents.ngcsacbc import NGCSACBCAgent
from agents.sharsa import SHARSAAgent

# Baselines (Model-based)
from agents.mbpo import MBPOAgent
from agents.mopo import MOPOAgent
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
    sharsa=SHARSAAgent,

    mbpo=MBPOAgent,
    mopo=MOPOAgent,
    combo=COMBOAgent,
    fmpc=FMPCAgent,
    leq=LEQAgent,
    mobile=MOBILEAgent,

    mac=MACAgent,
    mbrs_ac=ACMBRSAgent,
    mbfql=MBFQLAgent,
    model_ac=ACModelAgent,
)
