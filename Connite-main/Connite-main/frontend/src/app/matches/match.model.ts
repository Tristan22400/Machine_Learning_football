export class Match
{
    constructor(
        public id: number,
        public homeTeam: string,
        public awayTeam: string,
        public Season: string,
        public Date: string,
        public Status: string,
        public Winner: string,
        public Goal_Away: number,
        public Goal_Home: number,
        public Time: string,
    ) {}
}
