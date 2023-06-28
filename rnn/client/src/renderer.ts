import { Application, Graphics } from "pixi.js";
import Bug from "./bug";

export default class Renderer {
  private readonly app: Application;
  private readonly graphics: Graphics;

  constructor(private readonly bugs: ReadonlyArray<Bug>) {
    const appContainerElement = document.getElementById("app");
    if (!appContainerElement) {
      throw new Error("Couldn't find element #app");
    }

    this.app = new Application({ width: 800, height: 640 });
    appContainerElement.appendChild(this.app.view as HTMLCanvasElement);

    this.app.stage.eventMode = "static";
    this.app.stage.hitArea = this.app.screen;

    this.graphics = new Graphics();
    this.graphics.x = 0;
    this.graphics.y = 0;
    this.app.stage.addChild(this.graphics);

    this.app.ticker.add(this.tick.bind(this));
  }

  private tick(delta: number) {
    this.graphics.clear();
    this.graphics.lineStyle(2.0, "#4488DD", 1.0);
    this.graphics.beginFill("#88AADD");
    this.bugs.forEach((bug) => {
      bug.update(delta);
      this.graphics.drawCircle(
        (bug.getPosition().x + 0.5) * this.app.screen.width,
        (bug.getPosition().y + 0.5) * this.app.screen.height,
        bug.getSize() * Bug.MAX_SIZE_WORLD,
      );
    });
  }
}
