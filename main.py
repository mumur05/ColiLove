import pybullet as p
import pybullet_data
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from fpdf import FPDF 
import os 
from PIL import Image 

# --- PROPIEDADES DE MATERIALES (AHORA CON DENSIDAD 'rho') ---
MATERIALES = {
    "Acero":    {"rho": 7850, "young": 200e9, "rest": 0.6,  "fric": 0.5, "color": [0.6, 0.6, 0.7, 1]}, 
    "Madera":   {"rho": 700,  "young": 11e9,  "rest": 0.3,  "fric": 0.8, "color": [0.5, 0.3, 0.1, 1]}, 
    "Goma":     {"rho": 1100, "young": 0.05e9,"rest": 0.85, "fric": 1.0, "color": [0.1, 0.1, 0.1, 1]}, 
    "Oro":      {"rho": 19300,"young": 79e9,  "rest": 0.2,  "fric": 0.4, "color": [1.0, 0.8, 0.0, 1]}, 
    "Hormigón": {"rho": 2400, "young": 30e9,  "rest": 0.1,  "fric": 0.9, "color": [0.8, 0.8, 0.8, 1]},
    "Unicel":   {"rho": 30,   "young": 0.01e9,"rest": 0.5,  "fric": 0.5, "color": [0.9, 0.9, 0.9, 1]} # Para probar flotación
}

class AnalizadorColisiones:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Colisiones Físicas Pro v4")
        self.root.geometry("500x980")

        self.physicsClient = p.connect(p.GUI) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        p.resetDebugVisualizerCamera(cameraDistance=14, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,2])
        
        self.obj_A_id = None
        self.obj_B_id = None
        self.simulando = False
        self.pausa_colision = False
        self.cooldown_activo = False
        self.tiempo_pausa_inicio = 0
        self.tiempo_fin_pausa = 0

        self.trayectoria_ids = []
        self.prev_pos_A = None
        self.prev_pos_B = None
        
        self.historial = {"t": [], "E_total": []}
        self.historial_impactos = [] 
        self.tiempo_inicio = 0
        self.vel_prev_A = np.array([0,0,0])
        self.vel_prev_B = np.array([0,0,0])

        self.crear_interfaz()
        self.reset_entorno()
        self.loop_simulacion()

    def crear_interfaz(self):
        style = ttk.Style()
        style.configure("ObjA.TLabel", foreground="darkred", font=('Arial', 9, 'bold'))
        style.configure("ObjB.TLabel", foreground="darkblue", font=('Arial', 9, 'bold'))
        style.configure("Header.TLabel", font=('Arial', 10, 'bold'))

        tabs = ttk.Notebook(self.root)
        tabs.pack(fill='x', padx=10, pady=5)

        frame_conf = ttk.Frame(tabs)
        tabs.add(frame_conf, text="Configuración")

        # Posición
        frame_dist = ttk.LabelFrame(frame_conf, text="Posición Inicial")
        frame_dist.pack(fill='x', padx=5, pady=5)
        ttk.Label(frame_dist, text="Distancia entre objetos:").pack(side='left', padx=5)
        self.distancia_var = tk.DoubleVar(value=10.0) 
        self.scale_dist = ttk.Scale(frame_dist, from_=2, to=40, variable=self.distancia_var, orient='horizontal', command=self.update_dist_label)
        self.scale_dist.pack(side='left', fill='x', expand=True, padx=5)
        self.lbl_dist_val = ttk.Label(frame_dist, text="10.0 m", font=('Arial', 8, 'bold'))
        self.lbl_dist_val.pack(side='left', padx=5)
        
        # --- OBJETO A ---
        frame_A = ttk.LabelFrame(frame_conf, text="Objeto A (Izquierda)")
        frame_A.pack(fill='x', padx=5, pady=5)
        self.tipo_A = tk.StringVar(value="Esfera")
        self.mat_A = tk.StringVar(value="Acero")
        
        cb_tipo_A = ttk.Combobox(frame_A, textvariable=self.tipo_A, values=["Esfera", "Cubo"], width=8)
        cb_tipo_A.grid(row=0, column=0, padx=5)
        cb_tipo_A.bind("<<ComboboxSelected>>", lambda e: self.actualizar_masa_sugerida("A"))

        om_mat_A = ttk.OptionMenu(frame_A, self.mat_A, "Acero", *MATERIALES.keys(), command=lambda e: self.actualizar_masa_sugerida("A"))
        om_mat_A.grid(row=0, column=1)
        
        ttk.Label(frame_A, text="Masa (kg):").grid(row=0, column=2)
        self.masa_A = tk.DoubleVar(value=5.0)
        self.entry_masa_A = ttk.Entry(frame_A, textvariable=self.masa_A, width=8) # Guardamos referencia para actualizar
        self.entry_masa_A.grid(row=0, column=3)
        
        ttk.Label(frame_A, text="Velocidad Inicial:").grid(row=1, column=0, pady=(5,0))
        self.vel_A_input = tk.DoubleVar(value=15.0)
        scale_A = ttk.Scale(frame_A, from_=0, to=40, variable=self.vel_A_input, orient='horizontal', command=self.update_vel_labels)
        scale_A.grid(row=1, column=1, columnspan=2, sticky='ew', padx=5)
        self.lbl_val_vA = ttk.Label(frame_A, text="15.0 m/s", font=('Arial', 8))
        self.lbl_val_vA.grid(row=1, column=3)

        # --- OBJETO B ---
        frame_B = ttk.LabelFrame(frame_conf, text="Objeto B (Derecha)")
        frame_B.pack(fill='x', padx=5, pady=5)
        self.tipo_B = tk.StringVar(value="Esfera")
        self.mat_B = tk.StringVar(value="Madera")

        cb_tipo_B = ttk.Combobox(frame_B, textvariable=self.tipo_B, values=["Esfera", "Cubo"], width=8)
        cb_tipo_B.grid(row=0, column=0, padx=5)
        cb_tipo_B.bind("<<ComboboxSelected>>", lambda e: self.actualizar_masa_sugerida("B"))

        om_mat_B = ttk.OptionMenu(frame_B, self.mat_B, "Madera", *MATERIALES.keys(), command=lambda e: self.actualizar_masa_sugerida("B"))
        om_mat_B.grid(row=0, column=1)

        ttk.Label(frame_B, text="Masa (kg):").grid(row=0, column=2)
        self.masa_B = tk.DoubleVar(value=5.0)
        self.entry_masa_B = ttk.Entry(frame_B, textvariable=self.masa_B, width=8)
        self.entry_masa_B.grid(row=0, column=3)
        
        ttk.Label(frame_B, text="Velocidad Inicial:").grid(row=1, column=0, pady=(5,0))
        self.vel_B_input = tk.DoubleVar(value=15.0)
        scale_B = ttk.Scale(frame_B, from_=0, to=40, variable=self.vel_B_input, orient='horizontal', command=self.update_vel_labels)
        scale_B.grid(row=1, column=1, columnspan=2, sticky='ew', padx=5)
        self.lbl_val_vB = ttk.Label(frame_B, text="15.0 m/s", font=('Arial', 8))
        self.lbl_val_vB.grid(row=1, column=3)

        # Entorno
        frame_env = ttk.LabelFrame(frame_conf, text="Entorno")
        frame_env.pack(fill='x', padx=5, pady=5)
        self.entorno_var = tk.StringVar(value="Aire")
        ttk.OptionMenu(frame_env, self.entorno_var, "Aire", "Vacio", "Aire", "Agua").pack(side='left', padx=5)
        self.suelo_var = tk.StringVar(value="Pavimento")
        ttk.OptionMenu(frame_env, self.suelo_var, "Pavimento", "Hielo", "Pavimento", "Tierra", "Pasto").pack(side='left', padx=5)

        # Botones
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=10)
        ttk.Button(btn_frame, text="LANZAR / REINICIAR", command=self.iniciar_lanzamiento).pack(fill='x', pady=5)
        self.btn_detener = ttk.Button(btn_frame, text="DETENER SIMULACIÓN Y CREAR PDF", command=self.detener_simulacion_y_reporte, state='disabled') 
        self.btn_detener.pack(fill='x', pady=5)

        # Monitor
        self.frame_rt = ttk.LabelFrame(self.root, text="Datos en Tiempo Real")
        self.frame_rt.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(self.frame_rt, text="Velocidad A:", style="ObjA.TLabel").grid(row=0, column=0, sticky='e')
        self.lbl_rt_vA = ttk.Label(self.frame_rt, text="0.00")
        self.lbl_rt_vA.grid(row=0, column=1)
        
        ttk.Label(self.frame_rt, text="Velocidad B:", style="ObjB.TLabel").grid(row=0, column=2, sticky='e')
        self.lbl_rt_vB = ttk.Label(self.frame_rt, text="0.00")
        self.lbl_rt_vB.grid(row=0, column=3)

        ttk.Label(self.frame_rt, text="E. Cinética A:").grid(row=1, column=0, sticky='e')
        self.lbl_rt_kA = ttk.Label(self.frame_rt, text="0.00")
        self.lbl_rt_kA.grid(row=1, column=1)

        ttk.Label(self.frame_rt, text="E. Cinética B:").grid(row=1, column=2, sticky='e')
        self.lbl_rt_kB = ttk.Label(self.frame_rt, text="0.00")
        self.lbl_rt_kB.grid(row=1, column=3)

        # Reporte
        self.frame_rep = ttk.LabelFrame(self.root, text="Análisis de Impacto (Teórico)")
        self.frame_rep.pack(fill='x', padx=10, pady=5)
        self.lbl_fuerza = ttk.Label(self.frame_rep, text="Fuerza Impacto Promedio: - N")
        self.lbl_fuerza.pack(anchor='w')
        self.lbl_deformacion = ttk.Label(self.frame_rep, text="Deformación Est.: - mm", foreground="purple", font=('Arial', 10, 'bold'))
        self.lbl_deformacion.pack(anchor='w', pady=2)
        self.lbl_status = ttk.Label(self.frame_rep, text="Estado: Esperando...", foreground="grey")
        self.lbl_status.pack(anchor='center', pady=(5,0))

        # Gráfica
        plt.style.use('dark_background') 
        self.fig, self.ax = plt.subplots(figsize=(4, 2.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        self.ax.set_title("Energía Total (J)", color='white', fontsize=12)
        self.fig.set_facecolor('#1e1e1e') 
        self.canvas.draw()

        # Calcular masas iniciales
        self.actualizar_masa_sugerida("A")
        self.actualizar_masa_sugerida("B")

    def actualizar_masa_sugerida(self, obj):
        # Calcula la masa real basándose en la densidad y el volumen del objeto
        if obj == "A":
            mat_nombre = self.mat_A.get()
            tipo = self.tipo_A.get()
            target_var = self.masa_A
        else:
            mat_nombre = self.mat_B.get()
            tipo = self.tipo_B.get()
            target_var = self.masa_B
        
        rho = MATERIALES[mat_nombre]["rho"]
        radio = 0.5
        
        if tipo == "Esfera":
            volumen = (4/3) * np.pi * (radio**3)
        else: # Cubo
            volumen = (radio * 2) ** 3
            
        masa_real = rho * volumen
        target_var.set(round(masa_real, 2))

    def update_vel_labels(self, event=None):
        self.lbl_val_vA.config(text=f"{self.vel_A_input.get():.1f} m/s")
        self.lbl_val_vB.config(text=f"{self.vel_B_input.get():.1f} m/s")

    def update_dist_label(self, event=None):
        self.lbl_dist_val.config(text=f"{self.distancia_var.get():.1f} m")

    def get_params(self):
        d_map = {"Vacio": 0.0, "Aire": 1.225, "Agua": 1000.0}
        rho = d_map.get(self.entorno_var.get(), 1.225)
        s_map = {"Hielo": (0.02, 0.9), "Pavimento": (0.7, 0.6), "Tierra": (0.9, 0.3), "Pasto": (1.0, 0.1)}
        fric, rest = s_map.get(self.suelo_var.get(), (0.7, 0.6))
        return rho, fric, rest

    def crear_objeto(self, tipo, pos, masa, material_nombre):
        props = MATERIALES[material_nombre]
        color = props["color"]
        
        if tipo == "Esfera":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=color)
        else:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=color)
        
        body_id = p.createMultiBody(masa, col, vis, basePosition=pos)
        p.changeDynamics(body_id, -1, restitution=props["rest"], lateralFriction=props["fric"])
        return body_id

    def reset_entorno(self):
        p.resetSimulation()
        
        if self.entorno_var.get() == "Vacio":
            p.setGravity(0, 0, 0)
        else:
            p.setGravity(0, 0, -9.81)
            
        p.loadURDF("plane.urdf")
        _, fric, rest = self.get_params()
        p.changeDynamics(0, -1, lateralFriction=fric, restitution=rest)
        p.removeAllUserDebugItems()
        
        self.trayectoria_ids = []
        self.historial = {"t": [], "E_total": []}
        
        self.ax.clear()
        self.ax.set_title("Energía Total del Sistema (J)", color='white', fontsize=12)
        self.ax.set_xlabel("Tiempo (s)", color='gray') 
        self.ax.set_ylabel("Energía (J)", color='gray') 
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.tick_params(colors='white')
        self.canvas.draw()
        
        self.lbl_fuerza.config(text="Fuerza Impacto Promedio: - N")
        self.lbl_deformacion.config(text="Deformación Est.: - mm")

    def iniciar_lanzamiento(self):
        self.reset_entorno()
        self.historial_impactos = [] 
        
        dist = self.distancia_var.get()
        mitad_dist = dist / 2.0
        
        pos_A = [-mitad_dist, 0, 1]
        pos_B = [mitad_dist, 0, 1]
        
        self.obj_A_id = self.crear_objeto(self.tipo_A.get(), pos_A, self.masa_A.get(), self.mat_A.get())
        self.obj_B_id = self.crear_objeto(self.tipo_B.get(), pos_B, self.masa_B.get(), self.mat_B.get())
        
        vA_ini_mag = self.vel_A_input.get()
        vB_ini_mag = self.vel_B_input.get()
        
        vA_ini_vec = [vA_ini_mag, 0, 2]
        vB_ini_vec = [-vB_ini_mag, 0, 2]
        
        p.resetBaseVelocity(self.obj_A_id, linearVelocity=vA_ini_vec)
        p.resetBaseVelocity(self.obj_B_id, linearVelocity=vB_ini_vec)
        
        self.vel_prev_A = np.array(vA_ini_vec)
        self.vel_prev_B = np.array(vB_ini_vec)
        
        self.simulando = True
        self.pausa_colision = False
        self.cooldown_activo = False
        self.tiempo_inicio = time.time()
        self.prev_pos_A = pos_A
        self.prev_pos_B = pos_B
        self.lbl_status.config(text="Simulando...", foreground="green")
        self.btn_detener.config(state='enabled')

    def calcular_fisica_extra(self, uid):
        vel, _ = p.getBaseVelocity(uid)
        v_vec = np.array(vel)
        v_mag = np.linalg.norm(v_vec)
        
        # Obtener propiedades
        rho_fluido, _, _ = self.get_params()
        masa = p.getDynamicsInfo(uid, -1)[0]
        
        # Volumen aproximado (Esfera r=0.5)
        radio = 0.5
        volumen = (4/3) * np.pi * (radio**3) 
        
        # --- 1. ARRASTRE (DRAG) ---
        if v_mag > 0 and rho_fluido > 0:
            area_transversal = np.pi * (radio**2)
            cd = 0.47 # Coeficiente esfera
            
            fuerza_drag_mag = 0.5 * rho_fluido * (v_mag**2) * cd * area_transversal
            
            # Estabilidad numérica: limitar fuerza para no revertir movimiento en 1 frame
            dt = 1/240.0
            fuerza_maxima = (masa * v_mag) / dt
            if fuerza_drag_mag > fuerza_maxima:
                fuerza_drag_mag = fuerza_maxima * 0.9 
            
            fuerza_drag_vec = -(v_vec / v_mag) * fuerza_drag_mag
            p.applyExternalForce(uid, -1, fuerza_drag_vec, [0,0,0], p.LINK_FRAME)

        # --- 2. EMPUJE (FLOTABILIDAD) ---
        if rho_fluido > 0:
            # Principio de Arquímedes: Fuerza = peso del fluido desplazado
            fuerza_empuje_mag = rho_fluido * volumen * 9.81
            p.applyExternalForce(uid, -1, [0, 0, fuerza_empuje_mag], [0,0,0], p.LINK_FRAME)
            
        return v_vec, v_mag, 0.5 * masa * v_mag**2

    def calcular_deformacion_revisada(self, fuerza, material_nombre):
        E = MATERIALES[material_nombre]["young"]
        L = 0.5 
        A_eff = np.pi * (0.05**2) 
        
        if E < 1e6: E = 1e6 
        factor_dinamico = 1.5 
        deformacion_metros = (fuerza * L * factor_dinamico) / (A_eff * E)
        return deformacion_metros * 1000 

    def capturar_imagen_pybullet(self):
        pos_A, _ = p.getBasePositionAndOrientation(self.obj_A_id)
        pos_B, _ = p.getBasePositionAndOrientation(self.obj_B_id)
        
        target_pos = [(pos_A[0] + pos_B[0]) / 2.0, (pos_A[1] + pos_B[1]) / 2.0, 1.0]
        distance_zoom = 3.0 
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(target_pos, distance_zoom, 0, -30, 0, 2)
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
        
        width, height, rgb, _, _ = p.getCameraImage(800, 600, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img_array = np.reshape(rgb, (height, width, 4))[:,:,:3] 
        img = Image.fromarray(img_array.astype(np.uint8))
        
        img_filename = f"impacto_pybullet_{time.strftime('%Y%m%d_%H%M%S')}_{len(self.historial_impactos) + 1}.png"
        img.save(img_filename)
        return img_filename

    def procesar_impacto(self, id1, id2, mat1_name, mat2_name, tipo_impacto):
        m1 = p.getDynamicsInfo(id1, -1)[0] if id1 != 0 else 1e9 
        m2 = p.getDynamicsInfo(id2, -1)[0] if id2 != 0 else 1e9

        v1_vec_post, _ = p.getBaseVelocity(id1)
        v2_vec_post, _ = p.getBaseVelocity(id2)
        v1_vec_post = np.array(v1_vec_post)
        v2_vec_post = np.array(v2_vec_post)
        
        v1_vec_pre = np.array([0.0, 0.0, 0.0])
        if id1 == self.obj_A_id: v1_vec_pre = self.vel_prev_A
        elif id1 == self.obj_B_id: v1_vec_pre = self.vel_prev_B

        v2_vec_pre = np.array([0.0, 0.0, 0.0])
        if id2 == self.obj_A_id: v2_vec_pre = self.vel_prev_A
        elif id2 == self.obj_B_id: v2_vec_pre = self.vel_prev_B
            
        dt_impacto = 0.015 
        F_total = 0.0
        def_1 = 0.0
        def_2 = 0.0
        num_participantes = 0
        
        if id1 != 0:
            delta_v1 = np.linalg.norm(v1_vec_post - v1_vec_pre)
            F1 = (m1 * delta_v1) / dt_impacto
            def_1 = self.calcular_deformacion_revisada(F1, mat1_name)
            F_total += F1
            num_participantes += 1
        
        if id2 != 0:
            delta_v2 = np.linalg.norm(v2_vec_post - v2_vec_pre)
            F2 = (m2 * delta_v2) / dt_impacto
            def_2 = self.calcular_deformacion_revisada(F2, mat2_name)
            F_total += F2
            num_participantes += 1
            
        F_promedio = F_total / num_participantes if num_participantes > 0 else 0.0
        
        imagen_impacto_path = self.capturar_imagen_pybullet()
        
        datos_impacto = {
            "tiempo": time.time() - self.tiempo_inicio,
            "fuerza": F_promedio,
            "def_A": def_1 if id1 == self.obj_A_id else (def_2 if id2 == self.obj_A_id else 0.0), 
            "def_B": def_2 if id2 == self.obj_B_id else (def_1 if id1 == self.obj_B_id else 0.0),
            "tipo": tipo_impacto, 
            "imagen_path": imagen_impacto_path 
        }
        self.historial_impactos.append(datos_impacto)
        
        self.lbl_fuerza.config(text=f"Fuerza Impacto Promedio: {F_promedio:.2f} N ({tipo_impacto})")
        
        def_A_str = f"{datos_impacto['def_A']:.2f} mm" if datos_impacto['def_A'] > 0 else "N/A"
        def_B_str = f"{datos_impacto['def_B']:.2f} mm" if datos_impacto['def_B'] > 0 else "N/A"
        self.lbl_deformacion.config(text=f"Def A: {def_A_str} | Def B: {def_B_str}")

        self.lbl_status.config(text=f"!!! IMPACTO DETECTADO: {tipo_impacto} !!!", foreground="red")
        
        self.pausa_colision = True
        self.tiempo_pausa_inicio = time.time()

    def dibujar_trayectoria(self):
        pos_A, _ = p.getBasePositionAndOrientation(self.obj_A_id)
        pos_B, _ = p.getBasePositionAndOrientation(self.obj_B_id)
        
        if self.prev_pos_A is not None and np.linalg.norm(np.array(pos_A) - np.array(self.prev_pos_A)) > 0.05:
            p.addUserDebugLine(self.prev_pos_A, pos_A, lineColorRGB=MATERIALES[self.mat_A.get()]["color"][:3], lineWidth=2, lifeTime=0)
            self.prev_pos_A = pos_A
            
        if self.prev_pos_B is not None and np.linalg.norm(np.array(pos_B) - np.array(self.prev_pos_B)) > 0.05:
            p.addUserDebugLine(self.prev_pos_B, pos_B, lineColorRGB=MATERIALES[self.mat_B.get()]["color"][:3], lineWidth=2, lifeTime=0)
            self.prev_pos_B = pos_B

    def loop_simulacion(self):
        if self.simulando:
            if self.pausa_colision:
                if time.time() - self.tiempo_pausa_inicio > 3.0:
                    self.pausa_colision = False
                    self.cooldown_activo = True
                    self.tiempo_fin_pausa = time.time()
                    self.lbl_status.config(text="Reanudando...", foreground="orange")
                    vA_post, _ = p.getBaseVelocity(self.obj_A_id)
                    vB_post, _ = p.getBaseVelocity(self.obj_B_id)
                    self.vel_prev_A = np.array(vA_post)
                    self.vel_prev_B = np.array(vB_post)
                else:
                    self.root.update()
                    self.root.after(50, self.loop_simulacion)
                    return

            if self.cooldown_activo:
                if time.time() - self.tiempo_fin_pausa > 1.0:
                    self.cooldown_activo = False
                    self.lbl_status.config(text="Simulando...", foreground="green")

            vA_pre, _ = p.getBaseVelocity(self.obj_A_id)
            vB_pre, _ = p.getBaseVelocity(self.obj_B_id)
            self.vel_prev_A = np.array(vA_pre)
            self.vel_prev_B = np.array(vB_pre)
            
            self.calcular_fisica_extra(self.obj_A_id)
            self.calcular_fisica_extra(self.obj_B_id)
            p.stepSimulation()
            self.dibujar_trayectoria()

            _, vA, kA = self.calcular_fisica_extra(self.obj_A_id)
            _, vB, kB = self.calcular_fisica_extra(self.obj_B_id)
            
            self.lbl_rt_vA.config(text=f"{vA:.2f} m/s")
            self.lbl_rt_vB.config(text=f"{vB:.2f} m/s")
            self.lbl_rt_kA.config(text=f"{kA:.2f} J")
            self.lbl_rt_kB.config(text=f"{kB:.2f} J")
            
            self.historial["t"].append(time.time() - self.tiempo_inicio)
            self.historial["E_total"].append(kA + kB)
            
            if len(self.historial["t"]) % 20 == 0: 
                self.ax.clear()
                self.ax.plot(self.historial["t"], self.historial["E_total"], color='#ff00ff', linewidth=2, label="E Total") 
                self.ax.set_title(f"Energía Total: {(kA+kB):.1f} J", color='#ff00ff', fontsize=12)
                y_max = max(self.historial["E_total"]) * 1.1 if self.historial["E_total"] else 1200
                self.ax.set_ylim(0, max(100, y_max))
                self.ax.set_xlabel("Tiempo (s)", color='gray')
                self.ax.set_ylabel("Energía (J)", color='gray')
                self.ax.grid(True, linestyle='--', alpha=0.5)
                self.ax.tick_params(colors='white')
                self.canvas.draw_idle()

            if not self.cooldown_activo and not self.pausa_colision:
                impacto_detectado = False
                
                contactos_AB = p.getContactPoints(self.obj_A_id, self.obj_B_id)
                if contactos_AB:
                    self.procesar_impacto(self.obj_A_id, self.obj_B_id, self.mat_A.get(), self.mat_B.get(), "Colisión A vs B")
                    impacto_detectado = True
                
                if not impacto_detectado:
                    contactos_A_G = p.getContactPoints(self.obj_A_id, 0)
                    if contactos_A_G:
                        self.procesar_impacto(self.obj_A_id, 0, self.mat_A.get(), "Hormigón", "Impacto A vs Suelo")
                        impacto_detectado = True
                        
                if not impacto_detectado:
                    contactos_B_G = p.getContactPoints(self.obj_B_id, 0)
                    if contactos_B_G:
                        self.procesar_impacto(self.obj_B_id, 0, self.mat_B.get(), "Hormigón", "Impacto B vs Suelo")

        self.root.after(16, self.loop_simulacion)

    def detener_simulacion_y_reporte(self):
        temp_images_to_delete = [impacto['imagen_path'] for impacto in self.historial_impactos]
        if self.simulando:
            self.simulando = False
            self.lbl_status.config(text="Simulación Detenida. Generando Reporte...", foreground="blue")
            self.root.update()
            
            if self.historial_impactos:
                self.generar_pdf_reporte(temp_images_to_delete)
                self.lbl_status.config(text="Reporte PDF Creado con Éxito.", foreground="green")
            else:
                self.lbl_status.config(text="Simulación Detenida. No se detectaron impactos.", foreground="orange")
        self.btn_detener.config(state='disabled')

    def generar_pdf_reporte(self, temp_images_to_delete):
        plot_filename = "energia_total_plot_temp.png"
        self.fig.savefig(plot_filename)
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "REPORTE DE ANÁLISIS DE COLISIONES", 0, 1, "C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 5, f"Fecha del Reporte: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        pdf.ln(5)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "1. Configuración Inicial", 0, 1, "L")
        
        pdf.set_fill_color(220, 220, 220)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(90, 6, "Objeto A", 1, 0, 'L', 1)
        pdf.cell(90, 6, "Objeto B", 1, 1, 'L', 1)

        pdf.set_font("Arial", "", 10)
        mat_A_props = MATERIALES[self.mat_A.get()]
        mat_B_props = MATERIALES[self.mat_B.get()]
        
        pdf.cell(90, 5, f"  Material: {self.mat_A.get()} (E={mat_A_props['young']:.1e} Pa)", 1, 0, 'L')
        pdf.cell(90, 5, f"  Material: {self.mat_B.get()} (E={mat_B_props['young']:.1e} Pa)", 1, 1, 'L')
        pdf.cell(90, 5, f"  Masa: {self.masa_A.get():.1f} kg (R={mat_A_props['rest']:.2f})", 1, 0, 'L')
        pdf.cell(90, 5, f"  Masa: {self.masa_B.get():.1f} kg (R={mat_B_props['rest']:.2f})", 1, 1, 'L')
        pdf.cell(90, 5, f"  Velocidad Inicial: {self.vel_A_input.get():.1f} m/s", 1, 0, 'L')
        pdf.cell(90, 5, f"  Velocidad Inicial: {self.vel_B_input.get():.1f} m/s", 1, 1, 'L')
        
        pdf.set_font("Arial", "I", 9)
        pdf.cell(0, 5, f"Entorno: {self.entorno_var.get()} / Suelo: {self.suelo_var.get()} / Distancia: {self.distancia_var.get():.1f} m", 0, 1)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "2. Gráfica de Energía Total del Sistema", 0, 1, "L")
        pdf.image(plot_filename, x = 15, y = pdf.get_y(), w = 180) 
        pdf.ln(130) 

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"3. Historial Detallado de Impactos", 0, 1, "L")
        
        ANCHO_IMAGEN = 90
        ALTURA_IMAGEN = 67.5 
        
        for i, impacto in enumerate(self.historial_impactos):
            if pdf.get_y() + ALTURA_IMAGEN + 10 > (297 - pdf.b_margin): 
                pdf.add_page()
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "3. Historial Detallado de Impactos (Continuación)", 0, 1, "L")
            
            pdf.set_fill_color(150, 150, 200)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 7, f"Impacto Nro. {i + 1}: {impacto['tipo']} (Tiempo: {impacto['tiempo']:.2f} s)", 1, 1, 'L', 1)
            pdf.ln(1)
            y_start = pdf.get_y() 
            
            pdf.image(impacto['imagen_path'], x = 105, y = y_start, w = ANCHO_IMAGEN, h = ALTURA_IMAGEN)
            
            pdf.set_xy(15, y_start) 
            pdf.set_font("Arial", "B", 9)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(45, 6, "Variable", 1, 0, 'C', 1)
            pdf.cell(45, 6, "Valor", 1, 1, 'C', 1)
            
            pdf.set_font("Arial", "", 9)
            pdf.set_x(15) 
            pdf.cell(45, 5, "Fuerza Impacto Promedio:", 1, 0, 'L')
            pdf.cell(45, 5, f"{impacto['fuerza']:.2f} N", 1, 1, 'L')
            
            def_A_str = f"{impacto['def_A']:.2f} mm" if impacto['def_A'] > 0 else "N/A"
            pdf.set_x(15) 
            pdf.cell(45, 5, f"Deformación A ({self.mat_A.get()}):", 1, 0, 'L')
            pdf.cell(45, 5, def_A_str, 1, 1, 'L')
            
            def_B_str = f"{impacto['def_B']:.2f} mm" if impacto['def_B'] > 0 else "N/A"
            pdf.set_x(15) 
            pdf.cell(45, 5, f"Deformación B ({self.mat_B.get()}):", 1, 0, 'L')
            pdf.cell(45, 5, def_B_str, 1, 1, 'L')
            
            pdf.set_y(y_start + ALTURA_IMAGEN + 5) 
            pdf.ln(5)

        report_filename = f"Reporte_Colision_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(report_filename)
        
        os.remove(plot_filename)
        for img_path in temp_images_to_delete:
            if os.path.exists(img_path):
                os.remove(img_path)
        
        print(f"Reporte generado: {report_filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalizadorColisiones(root)      
    root.mainloop()
