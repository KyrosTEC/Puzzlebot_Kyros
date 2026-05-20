import time


class PIDController:
    """
    Controlador PID genérico con anti-windup y límite de salida.

    Uso:
        pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.1, output_min=-2.0, output_max=2.0)
        output = pid.compute(error)   # llamar a frecuencia constante

    Anti-windup: el integrador se congela cuando la salida está saturada.
    Derivativo: sobre el error (no sobre la medición) con filtro de primer orden.
    """

    def __init__(self, Kp, Ki, Kd,
                 output_min=-float('inf'),
                 output_max=float('inf'),
                 derivative_filter=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_min = output_min
        self.output_max = output_max
        self.derivative_filter = derivative_filter  # α filtro derivativo [0-1]

        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_deriv = 0.0
        self._last_time  = None

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_deriv = 0.0
        self._last_time  = None

    def compute(self, error):
        now = time.time()

        if self._last_time is None:
            dt = 0.02   # primer ciclo: asumir 50 Hz
        else:
            dt = now - self._last_time
            if dt <= 0 or dt > 0.5:
                dt = 0.02   # descartar dt inválido
        self._last_time = now

        # Proporcional
        P = self.Kp * error

        # Integral con anti-windup (solo integra si no estamos saturados)
        self._integral += error * dt
        I = self.Ki * self._integral

        # Derivativo con filtro EMA para reducir ruido
        raw_deriv = (error - self._prev_error) / dt
        D_filtered = (self.derivative_filter * raw_deriv +
                      (1.0 - self.derivative_filter) * self._prev_deriv)
        D = self.Kd * D_filtered

        self._prev_error = error
        self._prev_deriv = D_filtered

        output = P + I + D

        # Saturación + anti-windup: si saturamos, revertir la integración
        if output > self.output_max:
            self._integral -= error * dt   # revertir
            output = self.output_max
        elif output < self.output_min:
            self._integral -= error * dt
            output = self.output_min

        return output