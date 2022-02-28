export class EndSeason
{
    constructor(
        public Team: string,
        public Points: number,
        public Wins: number,
        public Draws: number,
        public Losses: number,
        public Goals_scored: number,
        public Goals_conceded: number,
        public Rank: number,

    ) {}
}
