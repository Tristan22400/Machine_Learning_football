import {Injectable} from '@angular/core';
import {HttpClient,} from '@angular/common/http';
import {API_URL} from '../env';
import {Rankings} from './rankings.model';

@Injectable()
export class StatApiService
{
    constructor(private http: HttpClient)
    {}

    getRankings(team: string)
    {   
        return this.http
            .get<Rankings>(`${API_URL}/rankings/${team}`)
    }

}
