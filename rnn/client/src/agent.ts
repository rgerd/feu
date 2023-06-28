export interface AgentControl {
    turn: number;
    shouldLunge: boolean;
}

export default class Agent {
    constructor() {
    }

    public getControl(): AgentControl {
        return {
            turn: Math.random() - 0.5,
            shouldLunge: Math.random() < 0.01
        };
    }
}