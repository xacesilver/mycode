import sys
import torch
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# Import your EXACT model from the file you uploaded
from Triton1_GPUTriton_fixed_v8 import TritonHierarchicalPlatform

class RealTritonDashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genuine Triton PyTorch Dashboard")
        self.resize(1200, 800)

        # 1. Initialize your EXACT PyTorch Model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lattice_shape = (128, 128)
        self.model = TritonHierarchicalPlatform(
            lattice_shape=self.lattice_shape, 
            device=self.device, 
            use_fixed_point=False
        )
        
        # FIX 1: Assign raw tensors directly to the buffers
        self.model.prev_syntax_graph = torch.zeros((1, 64, 64), device=self.device)
        self.model.relational_syntax_graph = torch.rand((1, 64, 64), device=self.device)
        self.model.spatial_routing_mask = torch.ones((1, *self.lattice_shape), device=self.device)
        
        # FIX 2: Explicitly trigger the platform reset to pre-allocate `state` and `V_B_ema` tensors 
        # so they aren't NoneType when the fast recurrent block is called.
        self.model.reset_platform(batch_size=1)
            
        self.I_bias = torch.zeros((1, *self.lattice_shape), dtype=torch.complex64, device=self.device)

        # 2. Setup the UI Layout
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QHBoxLayout(cw)

        # Left control panel
        control_panel = QtWidgets.QVBoxLayout()
        
        self.btn_teleology = QtWidgets.QCheckBox("Teleological Homeostasis")
        self.btn_identity = QtWidgets.QCheckBox("Identity Resonance")
        self.btn_shadow = QtWidgets.QCheckBox("Shadow Counterfactual")
        self.btn_auto = QtWidgets.QCheckBox("Autopoietic Synthesis")
        
        control_panel.addWidget(self.btn_teleology)
        control_panel.addWidget(self.btn_identity)
        control_panel.addWidget(self.btn_shadow)
        control_panel.addWidget(self.btn_auto)
        
        self.btn_pulse = QtWidgets.QPushButton("Inject Central Pulse")
        self.btn_pulse.clicked.connect(self.inject_pulse)
        control_panel.addWidget(self.btn_pulse)

        control_panel.addStretch()
        
        # Right Visualization Panel
        vis_layout = pg.GraphicsLayoutWidget()
        
        # Setup image items for your specific tensors
        self.img_state = pg.ImageItem()
        self.img_affect = pg.ImageItem()
        self.img_ennui = pg.ImageItem()
        self.img_omega = pg.ImageItem()

        p1 = vis_layout.addPlot(title="Layer 0 State (Amplitude)")
        p1.addItem(self.img_state)
        
        p2 = vis_layout.addPlot(title="Affective Valence")
        p2.addItem(self.img_affect)
        
        vis_layout.nextRow()
        
        p3 = vis_layout.addPlot(title="Cognitive Ennui")
        p3.addItem(self.img_ennui)
        
        p4 = vis_layout.addPlot(title="Volatility Omega")
        p4.addItem(self.img_omega)

        layout.addLayout(control_panel, 1)
        layout.addWidget(vis_layout, 4)

        # 3. Setup the live execution loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(50) # 20 FPS updates

    def inject_pulse(self):
        """Creates a real perturbation in your I_bias tensor"""
        center_x, center_y = self.lattice_shape[0]//2, self.lattice_shape[1]//2
        # Injecting a real spike into the exact PyTorch tensor
        self.I_bias[0, center_x-5:center_x+5, center_y-5:center_y+5] += 2.0 + 2.0j

    def update_simulation(self):
        """Runs your EXACT forward block and pulls the real tensors"""
        
        # Run 1 cycle of your exact code using the UI checkbox states
        with torch.no_grad():
            self.model.forward_fast_recurrent_block(
                I_bias=self.I_bias, 
                cycles=1, 
                teleological_homeostasis=self.btn_teleology.isChecked(),
                identity_resonance=self.btn_identity.isChecked(),
                shadow_counterfactual=self.btn_shadow.isChecked(),
                autopoietic_synthesis=self.btn_auto.isChecked()
            )
            
            # Decay the bias slightly over time
            self.I_bias *= 0.95

            # Pull the EXACT PyTorch tensors to the CPU and render them
            # Layer 0 State Amplitude (taking the B channel index 1)
            state_amp = torch.abs(self.model.layer0.state[0, ..., 1]).cpu().numpy()
            
            # Metacognitive telemetry
            affect = self.model.affective_valence[0].cpu().numpy()
            ennui = self.model.cognitive_ennui[0].cpu().numpy()
            omega = self.model.volatility_omega[0].cpu().numpy()

        # Update the UI images
        self.img_state.setImage(state_amp, autoLevels=True)
        self.img_affect.setImage(affect, autoLevels=True)
        self.img_ennui.setImage(ennui, autoLevels=True)
        self.img_omega.setImage(omega, autoLevels=True)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = RealTritonDashboard()
    window.show()
    sys.exit(app.exec_())
