import Agent from "./agent";

// https://easings.net/#easeInOutCirc
function easeInOutCirc(x: number): number {
  return x < 0.5
    ? (1 - Math.sqrt(1 - Math.pow(2 * x, 2))) / 2
    : (Math.sqrt(1 - Math.pow(-2 * x + 2, 2)) + 1) / 2;
}

export default class Bug {
  public static readonly MAX_SIZE_WORLD = 32;
  private static readonly MAX_TURN_SPEED = 1.0;
  private static readonly MOVE_SPEED = 0.001;
  private static readonly LUNGE_DISTANCE = 0.04;
  private static readonly LUNGE_SPEED = 0.01;

  private position = { x: 0, y: 0 };
  private direction = 0;
  private turn = 0;
  private size = 0.5; // (0.0 - 1.0)
  private lunge = 0.0; // (0.0 - 1.0)
  private lungeTrajectory: {
    start: { x: number; y: number };
    vector: { x: number; y: number };
  } | null = null;

  constructor(private readonly agent: Agent, initX: number, initY: number) {
    this.position = { x: initX, y: initY };
  }

  public update(delta: number): void {
    const control = this.agent.getControl();
    this.turn = control.turn;
    if (control.shouldLunge) {
      this.tryLunge();
    }

    this.direction += this.turn * delta;

    if (this.lungeTrajectory) {
      const ease = easeInOutCirc(this.lunge);
      this.position = {
        x: this.lungeTrajectory.vector.x * ease + this.lungeTrajectory.start.x,
        y: this.lungeTrajectory.vector.y * ease + this.lungeTrajectory.start.y,
      };
      this.lunge += delta * Bug.LUNGE_SPEED;
      if (this.lunge >= 1.0) {
        this.lunge = 0.0;
        this.lungeTrajectory = null;
      }
    } else {
      this.position.x += Math.cos(this.direction) * delta * Bug.MOVE_SPEED;
      this.position.y += Math.sin(this.direction) * delta * Bug.MOVE_SPEED;
    }
  }

  public tryLunge() {
    if (!this.lungeTrajectory && this.size > 0.1) {
      this.lunge = 0.0;
      this.size *= 0.85;
      this.lungeTrajectory = {
        start: { x: this.position.x, y: this.position.y },
        vector: {
          x: Math.cos(this.direction) * Bug.LUNGE_DISTANCE * (1.5 - this.size),
          y: Math.sin(this.direction) * Bug.LUNGE_DISTANCE * (1.5 - this.size),
        },
      };
    }
  }

  public getPosition(): { x: number; y: number } {
    return this.position;
  }

  public getSize(): number {
    return this.size;
  }
}
