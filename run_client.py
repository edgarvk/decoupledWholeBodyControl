import time
import cv2
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from gr00t.policy.server_client import PolicyClient


# ============================================================
# CONFIGURACIÓN
# ============================================================

NETWORK_INTERFACE = "eth0"

POLICY_HOST = "127.0.0.1"
POLICY_PORT = 5555

TASK_INSTRUCTION = (
    "Pick up the red mug and place it on the table"
)

# Seguridad: solo inferencia y registro de acciones.
ENABLE_PHYSICAL_CONTROL = False

# No se permite enviar acciones hasta implementar el controlador
# y validar las modalidades del checkpoint.
CONTROL_VALIDATED = False

# Estado DDS del G1
latest_low_state = None


# ============================================================
# ESTADO DEL G1
# ============================================================

def low_state_callback(msg):
    global latest_low_state
    latest_low_state = msg


def initialize_state_subscriber():
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    subscriber = ChannelSubscriber(
        "rt/lowstate",
        LowState_,
    )
    subscriber.Init(low_state_callback, 10)

    return subscriber


def read_g1_arm_positions():
    """
    Lee los 14 motores de brazo del estado del G1.

    Índices de los 29 motores principales:
      15:22 brazo izquierdo
      22:29 brazo derecho

    Verifica el orden contra la versión de SDK y el modelo
    concreto del G1 antes de utilizar estos valores.
    """

    if latest_low_state is None:
        raise RuntimeError("No se ha recibido rt/lowstate.")

    motors = latest_low_state.motor_state

    if len(motors) < 29:
        raise RuntimeError(
            f"Estado incompleto: {len(motors)} motores."
        )

    left_arm = np.array(
        [motors[i].q for i in range(15, 22)],
        dtype=np.float32,
    )
    right_arm = np.array(
        [motors[i].q for i in range(22, 29)],
        dtype=np.float32,
    )

    if not (
        np.all(np.isfinite(left_arm))
        and np.all(np.isfinite(right_arm))
    ):
        raise RuntimeError("Estado articular no válido.")

    return left_arm, right_arm


# ============================================================
# ADAPTADORES DE HARDWARE PENDIENTES
# ============================================================

def read_g1_integrated_camera():
    """
    Debe devolver un frame BGR de la cámara integrada del G1.

    Integra aquí la API de cámara correspondiente a tu versión
    del SDK de Unitree. No utilizar VideoClient del Go2.

    La documentación del SDK incluye un ejemplo específico:
    example/front_camera/camera_opencv.py
    """

    raise NotImplementedError(
        "Integra el lector de la cámara frontal del G1."
    )


def read_dex3_hand_positions():
    """
    Debe devolver el estado real de las manos Dex3.

    Formato requerido por este adaptador:
        {
            'left_hand':  np.ndarray(shape=(7,), dtype=np.float32),
            'right_hand': np.ndarray(shape=(7,), dtype=np.float32)
        }

    Los vectores son ilustrativos en cuanto a dimensiones:
    confirma el número, orden y unidades de articulaciones
    en la configuración de tu checkpoint y en el SDK Dex3.

    No sustituir por ceros ni reutilizar el estado de los brazos.
    """

    raise NotImplementedError(
        "Implementa la lectura DDS de ambas manos Dex3."
    )


# ============================================================
# GR00T: INFERENCIA
# ============================================================

def initialize_policy():
    policy = PolicyClient(
        host=POLICY_HOST,
        port=POLICY_PORT,
        timeout_ms=15000,
        strict=True,
    )

    if not policy.ping():
        raise RuntimeError(
            "El servidor GR00T no responde."
        )

    modality = policy.get_modality_config()

    print("\n=== MODALIDADES DEL CHECKPOINT ===")

    for name, config in modality.items():
        print(
            name,
            "keys =", getattr(config, "modality_keys", None),
            "horizon =", getattr(config, "delta_indices", None),
        )

    return policy, modality


def prepare_observation(frame_bgr, modality):
    left_arm, right_arm = read_g1_arm_positions()
    hands = read_dex3_hand_positions()

    # Las claves de vídeo se consultan al modelo, no se inventan.
    expected_cameras = list(
        modality["video"].modality_keys
    )

    if len(expected_cameras) != 1:
        raise RuntimeError(
            "El checkpoint espera estas cámaras: "
            f"{expected_cameras}. "
            "Una sola cámara no puede satisfacer automáticamente "
            "una configuración multicámara."
        )

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Ajusta el tamaño al procesador del checkpoint.
    frame_rgb = cv2.resize(
        frame_rgb,
        (224, 224),
        interpolation=cv2.INTER_AREA,
    )

    video = {
        expected_cameras[0]: frame_rgb[None, None, ...]
    }

    # REAL_G1: confirmar los nombres contra modality_config.
    state_values = {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_hand": hands["left_hand"],
        "right_hand": hands["right_hand"],
    }

    expected_states = list(
        modality["state"].modality_keys
    )

    missing = [
        key for key in expected_states
        if key not in state_values
    ]

    if missing:
        raise RuntimeError(
            "Faltan estados requeridos por el checkpoint: "
            f"{missing}. No se generará una observación incompleta."
        )

    state = {
        key: np.asarray(
            state_values[key],
            dtype=np.float32,
        )[None, None, ...]
        for key in expected_states
    }

    # Clave de lenguaje utilizada en configuraciones REAL_G1.
    language = {
        "annotation.human.task_description": [
            [TASK_INSTRUCTION]
        ]
    }

    return {
        "video": video,
        "state": state,
        "language": language,
    }


def infer_actions(policy, observation):
    action, info = policy.get_action(observation)

    if not isinstance(action, dict) or not action:
        raise RuntimeError("GR00T devolvió una acción vacía o inválida.")

    print("\n=== SALIDA GR00T ===")

    for name, value in action.items():
        arr = np.asarray(value)

        if not np.all(np.isfinite(arr)):
            raise RuntimeError(
                f"La salida {name} contiene NaN o Inf."
            )

        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")

    return action, info


# ============================================================
# CONTROL FÍSICO: BLOQUEADO
# ============================================================

def send_actions_to_g1(action):
    """
    No enviar directamente la salida REAL_G1 a rt/lowcmd.

    La política puede producir acciones relativas de espacio
    cartesiano y articulares. Se necesita un adaptador validado
    que interprete cada clave según el contrato del checkpoint,
    resuelva la cinemática inversa cuando corresponda y aplique
    límites y control de seguridad.
    """

    if not ENABLE_PHYSICAL_CONTROL or not CONTROL_VALIDATED:
        raise RuntimeError(
            "Control físico bloqueado: falta validar el adaptador."
        )

    raise NotImplementedError(
        "Implementa aquí el controlador validado del G1 y Dex3."
    )


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

def main():
    subscriber = None
    cap = None

    try:
        subscriber = initialize_state_subscriber()

        print("Esperando rt/lowstate...")

        while latest_low_state is None:
            time.sleep(0.1)

        policy, modality = initialize_policy()

        # Solo se utiliza VideoCapture si la cámara está expuesta
        # como dispositivo local. La cámara integrada por red
        # necesita el adaptador específico del SDK.
        frame = read_g1_integrated_camera()

        print("Iniciando inferencia REAL_G1.")

        while True:
            frame = read_g1_integrated_camera()

            if frame is None:
                print("No se recibió imagen.")
                time.sleep(0.05)
                continue

            cv2.imshow("G1 - Cámara", frame)

            observation = prepare_observation(
                frame,
                modality,
            )

            action, info = infer_actions(
                policy,
                observation,
            )

            # Las acciones se registran, no se ejecutan.
            if ENABLE_PHYSICAL_CONTROL:
                send_actions_to_g1(action)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupción solicitada.")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()
        print("Inferencia finalizada.")


if __name__ == "__main__":
    main()