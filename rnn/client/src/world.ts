import Agent from "./agent";
import Bug from "./bug";
import Renderer from "./renderer";

export default class World {
  private readonly bugs = new Array<Bug>();
  private readonly renderer: Renderer;

  constructor() {
    for (let i = 0; i < 50; i++) {
      this.bugs.push(
        new Bug(new Agent(), Math.random() - 0.5, Math.random() - 0.5),
      );
    }
    this.renderer = new Renderer(this.bugs);
  }
}
